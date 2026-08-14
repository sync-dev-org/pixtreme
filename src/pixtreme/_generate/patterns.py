"""GPU-native image generators with explicit geometry and broadcast-pattern contracts."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import cast

import cupy as cp
import numpy as np

from pixtreme._core import validation as _validation
from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import _COLORSPACE_TOKENS, _GAMMA_TOKENS, Frame
from pixtreme._core.validation import (
    _finite_pair,
    _finite_real,
    _normalized_closed_token,
    _positive_real,
    _positive_scalar_or_pair,
)
from pixtreme._core.vocabulary import (
    _COLOR_BARS_OUTPUT_TOKENS,
    _COLOR_BARS_STANDARD_TOKENS,
    _GENERATOR_KIND_TOKENS,
)
from pixtreme._draw.shapes import _AA_TOKENS

_KIND_TOKENS = _GENERATOR_KIND_TOKENS
_STANDARD_TOKENS = _COLOR_BARS_STANDARD_TOKENS
_OUTPUT_TOKENS = _COLOR_BARS_OUTPUT_TOKENS
_GENERATOR_BLOCK = (16, 16)
_GEOMETRY_RAMP_LINEAR = 0
_GEOMETRY_RAMP_RADIAL = 1
_GEOMETRY_GRID = 2
_GEOMETRY_CHECKERBOARD = 3
_HOST_ARRAY_WHY = "generator geometry and color inputs must be convertible to a regular host array"
_HOST_ARRAY_HOW = "pass a sequence, NumPy array, or CuPy array with a regular numeric shape"
_CELL_WHY = "cell must be one positive size or a positive (width, height) pair"
_CELL_HOW = "pass a finite positive real or a two-element finite positive real sequence"

_GEOMETRY_KERNEL_SOURCE = r"""
__device__ float pixtreme_generate_clamp01(const float value) {
    return value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
}

__device__ float pixtreme_generate_mod(const float value, const float period) {
    return value - floorf(value / period) * period;
}

__device__ float pixtreme_grid_distance(
    const float x,
    const float y,
    const float cell_x,
    const float cell_y,
    const float line_width,
    const float offset_x,
    const float offset_y
) {
    const float phase_x =
        pixtreme_generate_mod(x - offset_x + 0.5f * cell_x, cell_x) - 0.5f * cell_x;
    const float phase_y =
        pixtreme_generate_mod(y - offset_y + 0.5f * cell_y, cell_y) - 0.5f * cell_y;
    return fminf(fabsf(phase_x), fabsf(phase_y)) - 0.5f * line_width;
}

__device__ int pixtreme_checker_first(
    const float x,
    const float y,
    const float cell_x,
    const float cell_y,
    const float offset_x,
    const float offset_y
) {
    const long long cell_index_x = (long long)floorf((x - offset_x) / cell_x);
    const long long cell_index_y = (long long)floorf((y - offset_y) / cell_y);
    return ((cell_index_x + cell_index_y) & 1LL) == 0LL;
}

__device__ float pixtreme_checker_boundary_distance(
    const float x,
    const float y,
    const float cell_x,
    const float cell_y,
    const float offset_x,
    const float offset_y
) {
    const float phase_x = pixtreme_generate_mod(x - offset_x, cell_x);
    const float phase_y = pixtreme_generate_mod(y - offset_y, cell_y);
    return fminf(
        fminf(phase_x, cell_x - phase_x),
        fminf(phase_y, cell_y - phase_y)
    );
}

__device__ float pixtreme_grid_coverage(
    const float x,
    const float y,
    const float cell_x,
    const float cell_y,
    const float line_width,
    const float offset_x,
    const float offset_y,
    const int aa
) {
    if (aa == 2) {
        return pixtreme_grid_distance(
            x,
            y,
            cell_x,
            cell_y,
            line_width,
            offset_x,
            offset_y
        ) <= 0.0f ? 1.0f : 0.0f;
    }
    if (aa == 0) {
        return pixtreme_generate_clamp01(
            0.5f - pixtreme_grid_distance(
                x,
                y,
                cell_x,
                cell_y,
                line_width,
                offset_x,
                offset_y
            )
        );
    }

    const float offsets[4] = {-0.375f, -0.125f, 0.125f, 0.375f};
    float coverage = 0.0f;
    #pragma unroll
    for (int sample_y = 0; sample_y < 4; ++sample_y) {
        #pragma unroll
        for (int sample_x = 0; sample_x < 4; ++sample_x) {
            coverage += pixtreme_grid_distance(
                x + offsets[sample_x],
                y + offsets[sample_y],
                cell_x,
                cell_y,
                line_width,
                offset_x,
                offset_y
            ) <= 0.0f ? 1.0f : 0.0f;
        }
    }
    return coverage * (1.0f / 16.0f);
}

__device__ float pixtreme_checker_coverage(
    const float x,
    const float y,
    const float cell_x,
    const float cell_y,
    const float offset_x,
    const float offset_y,
    const int aa
) {
    if (aa == 2) {
        return pixtreme_checker_first(
            x,
            y,
            cell_x,
            cell_y,
            offset_x,
            offset_y
        ) ? 1.0f : 0.0f;
    }
    if (aa == 0) {
        const float distance = pixtreme_checker_boundary_distance(
            x,
            y,
            cell_x,
            cell_y,
            offset_x,
            offset_y
        );
        return pixtreme_generate_clamp01(
            pixtreme_checker_first(
                x,
                y,
                cell_x,
                cell_y,
                offset_x,
                offset_y
            ) ? 0.5f + distance : 0.5f - distance
        );
    }

    const float offsets[4] = {-0.375f, -0.125f, 0.125f, 0.375f};
    float coverage = 0.0f;
    #pragma unroll
    for (int sample_y = 0; sample_y < 4; ++sample_y) {
        #pragma unroll
        for (int sample_x = 0; sample_x < 4; ++sample_x) {
            coverage += pixtreme_checker_first(
                x + offsets[sample_x],
                y + offsets[sample_y],
                cell_x,
                cell_y,
                offset_x,
                offset_y
            ) ? 1.0f : 0.0f;
        }
    }
    return coverage * (1.0f / 16.0f);
}

extern "C" __global__ void pixtreme_generate_geometry(
    float* __restrict__ output,
    const float* __restrict__ first_color,
    const float* __restrict__ second_color,
    const long long width,
    const long long height,
    const long long channel_count,
    const int geometry,
    const int aa,
    const float parameter_0,
    const float parameter_1,
    const float parameter_2,
    const float parameter_3,
    const float parameter_4
) {
    const long long pixel_x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long pixel_y = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    if (pixel_x >= width || pixel_y >= height) {
        return;
    }

    const float x = (float)pixel_x + 0.5f;
    const float y = (float)pixel_y + 0.5f;
    float mix = 0.0f;
    if (geometry == 0 || geometry == 1) {
        const float delta_x = parameter_2 - parameter_0;
        const float delta_y = parameter_3 - parameter_1;
        if (geometry == 0) {
            mix = (
                (x - parameter_0) * delta_x +
                (y - parameter_1) * delta_y
            ) / (delta_x * delta_x + delta_y * delta_y);
        } else {
            mix = hypotf(x - parameter_0, y - parameter_1) / hypotf(delta_x, delta_y);
        }
        mix = pixtreme_generate_clamp01(mix);
    } else if (geometry == 2) {
        mix = pixtreme_grid_coverage(
            x,
            y,
            parameter_0,
            parameter_1,
            parameter_2,
            parameter_3,
            parameter_4,
            aa
        );
    } else {
        mix = pixtreme_checker_coverage(
            x,
            y,
            parameter_0,
            parameter_1,
            parameter_2,
            parameter_3,
            aa
        );
    }

    const long long output_offset = (pixel_y * width + pixel_x) * channel_count;
    for (long long channel = 0; channel < channel_count; ++channel) {
        if (geometry == 0 || geometry == 1) {
            output[output_offset + channel] =
                first_color[channel] * (1.0f - mix) + second_color[channel] * mix;
        } else {
            output[output_offset + channel] =
                second_color[channel] + (first_color[channel] - second_color[channel]) * mix;
        }
    }
}
"""

_COLOR_BARS_KERNEL_SOURCE = r"""
__device__ long long pixtreme_scaled_boundary(
    const long long reference,
    const long long size,
    const long long reference_size
) {
    return (reference * size + reference_size / 2LL) / reference_size;
}

__device__ int pixtreme_region(
    const long long coordinate,
    const long long* __restrict__ reference_boundaries,
    const int boundary_count,
    const long long size,
    const long long reference_size
) {
    for (int index = 0; index < boundary_count; ++index) {
        if (
            coordinate <
            pixtreme_scaled_boundary(reference_boundaries[index], size, reference_size)
        ) {
            return index;
        }
    }
    return boundary_count;
}

__device__ int pixtreme_scaled_ramp_code(
    const long long coordinate,
    const long long start,
    const long long end,
    const int first_code,
    const int last_code
) {
    const long long count = end - start;
    if (count <= 1LL) {
        return first_code;
    }
    const long long local = coordinate - start;
    const long long code_span = (long long)last_code - first_code;
    return first_code + (int)((local * code_span + (count - 1LL) / 2LL) / (count - 1LL));
}

__device__ void pixtreme_rgb(
    int* __restrict__ result,
    const int red,
    const int green,
    const int blue
) {
    result[0] = red;
    result[1] = green;
    result[2] = blue;
}

__device__ void pixtreme_primary_bar(
    int* __restrict__ result,
    const int index,
    const int high,
    const int low
) {
    if (index == 0) {
        pixtreme_rgb(result, high, high, high);
    } else if (index == 1) {
        pixtreme_rgb(result, high, high, low);
    } else if (index == 2) {
        pixtreme_rgb(result, low, high, high);
    } else if (index == 3) {
        pixtreme_rgb(result, low, high, low);
    } else if (index == 4) {
        pixtreme_rgb(result, high, low, high);
    } else if (index == 5) {
        pixtreme_rgb(result, high, low, low);
    } else if (index == 6) {
        pixtreme_rgb(result, low, low, high);
    } else {
        pixtreme_rgb(result, low, low, low);
    }
}

__device__ void pixtreme_std_b28(
    int* __restrict__ result,
    const long long x,
    const long long y,
    const long long width,
    const long long height
) {
    const long long y_boundaries[3] = {630, 720, 810};
    const int row = pixtreme_region(y, y_boundaries, 3, height, 1080);
    if (row == 0) {
        const long long boundaries[8] = {240, 445, 651, 857, 1063, 1269, 1475, 1680};
        const int region = pixtreme_region(x, boundaries, 8, width, 1920);
        if (region == 0 || region == 8) {
            pixtreme_rgb(result, 414, 414, 414);
        } else {
            pixtreme_primary_bar(result, region - 1, 721, 64);
        }
        return;
    }
    if (row == 1) {
        const long long boundaries[2] = {240, 1680};
        const int region = pixtreme_region(x, boundaries, 2, width, 1920);
        if (region == 0) {
            pixtreme_rgb(result, 64, 940, 940);
        } else if (region == 1) {
            pixtreme_rgb(result, 721, 721, 721);
        } else {
            pixtreme_rgb(result, 64, 64, 940);
        }
        return;
    }
    if (row == 2) {
        const long long boundaries[4] = {240, 445, 1475, 1680};
        const int region = pixtreme_region(x, boundaries, 4, width, 1920);
        if (region == 0) {
            pixtreme_rgb(result, 940, 940, 64);
        } else if (region == 1) {
            pixtreme_rgb(result, 64, 64, 64);
        } else if (region == 2) {
            const long long start = pixtreme_scaled_boundary(445, width, 1920);
            const long long end = pixtreme_scaled_boundary(1475, width, 1920);
            const int code = pixtreme_scaled_ramp_code(x, start, end, 64, 940);
            pixtreme_rgb(result, code, code, code);
        } else if (region == 3) {
            pixtreme_rgb(result, 940, 940, 940);
        } else {
            pixtreme_rgb(result, 940, 64, 64);
        }
        return;
    }

    const long long boundaries[10] = {
        240, 549, 960, 1131, 1200, 1268, 1337, 1405, 1474, 1680
    };
    const int region = pixtreme_region(x, boundaries, 10, width, 1920);
    const int codes[11] = {195, 64, 940, 64, 46, 64, 82, 64, 99, 64, 195};
    pixtreme_rgb(result, codes[region], codes[region], codes[region]);
}

__device__ void pixtreme_bt2111_equivalent(
    int* __restrict__ result,
    const int standard,
    const int index
) {
    if (standard == 2) {
        const int values[18] = {
            713, 719, 316, 538, 709, 718, 512, 706, 296,
            651, 286, 705, 639, 269, 164, 227, 147, 702
        };
        pixtreme_rgb(
            result,
            values[index * 3],
            values[index * 3 + 1],
            values[index * 3 + 2]
        );
    } else if (standard == 3) {
        const int values[18] = {
            568, 571, 381, 484, 566, 571, 474, 564, 368,
            536, 361, 564, 530, 350, 256, 317, 236, 562
        };
        pixtreme_rgb(
            result,
            values[index * 3],
            values[index * 3 + 1],
            values[index * 3 + 2]
        );
    } else {
        const int values[18] = {
            589, 592, 370, 491, 586, 592, 478, 584, 355,
            551, 347, 584, 544, 334, 225, 296, 201, 582
        };
        pixtreme_rgb(
            result,
            values[index * 3],
            values[index * 3 + 1],
            values[index * 3 + 2]
        );
    }
}

__device__ void pixtreme_bt2111(
    int* __restrict__ result,
    const long long x,
    const long long y,
    const long long width,
    const long long height,
    const int standard
) {
    const int full = standard == 4;
    const int low = full ? 0 : 64;
    const int top_high = full ? 1023 : 940;
    const int main_high = standard == 2 ? 721 : (standard == 3 ? 572 : 593);
    const int grey = full ? 409 : 414;
    const long long y_boundaries[4] = {90, 630, 720, 810};
    const int row = pixtreme_region(y, y_boundaries, 4, height, 1080);

    if (row == 0 || row == 1) {
        const long long boundaries[8] = {240, 446, 652, 858, 1062, 1268, 1474, 1680};
        const int region = pixtreme_region(x, boundaries, 8, width, 1920);
        if (region == 0 || region == 8) {
            pixtreme_rgb(result, grey, grey, grey);
        } else {
            pixtreme_primary_bar(
                result,
                region - 1,
                row == 0 ? top_high : main_high,
                low
            );
        }
        return;
    }

    if (row == 2) {
        const long long boundaries[14] = {
            240, 446, 549, 652, 755, 858, 960,
            1062, 1165, 1268, 1371, 1474, 1577, 1680
        };
        const int region = pixtreme_region(x, boundaries, 14, width, 1920);
        if (region == 0 || region == 14) {
            pixtreme_rgb(result, main_high, main_high, main_high);
        } else {
            const int narrow_codes[13] = {
                4, 64, 152, 239, 327, 414, 502, 590, 677, 765, 852, 940, 1019
            };
            const int full_codes[13] = {
                0, 0, 102, 205, 307, 409, 512, 614, 716, 818, 921, 1023, 1023
            };
            const int code = full ? full_codes[region - 1] : narrow_codes[region - 1];
            pixtreme_rgb(result, code, code, code);
        }
        return;
    }

    if (row == 3) {
        const long long black_end = pixtreme_scaled_boundary(240, width, 1920);
        if (x < black_end) {
            pixtreme_rgb(result, low, low, low);
            return;
        }
        const long long ramp_start = pixtreme_scaled_boundary(full ? 791 : 799, width, 1920);
        const long long ramp_end = pixtreme_scaled_boundary(1813, width, 1920);
        if (x < ramp_start) {
            const int code = full ? 0 : 4;
            pixtreme_rgb(result, code, code, code);
        } else if (x < ramp_end) {
            const int code = pixtreme_scaled_ramp_code(
                x,
                ramp_start,
                ramp_end,
                full ? 1 : 5,
                full ? 1022 : 1018
            );
            pixtreme_rgb(result, code, code, code);
        } else {
            const int code = full ? 1023 : 1019;
            pixtreme_rgb(result, code, code, code);
        }
        return;
    }

    const long long boundaries[14] = {
        80, 160, 240, 376, 446, 514, 584,
        652, 722, 960, 1398, 1680, 1760, 1840
    };
    const int region = pixtreme_region(x, boundaries, 14, width, 1920);
    if (region < 3) {
        pixtreme_bt2111_equivalent(result, standard, region);
    } else if (region == 3 || region == 9 || region == 11) {
        pixtreme_rgb(result, low, low, low);
    } else if (region == 4) {
        const int code = full ? 0 : 48;
        pixtreme_rgb(result, code, code, code);
    } else if (region == 5 || region == 7) {
        pixtreme_rgb(result, low, low, low);
    } else if (region == 6) {
        const int code = full ? 20 : 80;
        pixtreme_rgb(result, code, code, code);
    } else if (region == 8) {
        const int code = full ? 41 : 99;
        pixtreme_rgb(result, code, code, code);
    } else if (region == 10) {
        pixtreme_rgb(result, main_high, main_high, main_high);
    } else {
        pixtreme_bt2111_equivalent(result, standard, region - 9);
    }
}

__device__ void pixtreme_full_field(
    int* __restrict__ result,
    const long long x,
    const long long width,
    const int standard
) {
    const long long boundaries[7] = {240, 480, 720, 960, 1200, 1440, 1680};
    const int region = pixtreme_region(x, boundaries, 7, width, 1920);
    if (standard == 5) {
        pixtreme_primary_bar(result, region, 940, 64);
    } else if (region == 0) {
        pixtreme_rgb(result, 940, 940, 940);
    } else {
        pixtreme_primary_bar(result, region, 721, 64);
    }
}

__device__ void pixtreme_color_bar_code(
    int* __restrict__ result,
    const long long x,
    const long long y,
    const long long width,
    const long long height,
    const int standard
) {
    if (standard <= 1) {
        pixtreme_std_b28(result, x, y, width, height);
    } else if (standard <= 4) {
        pixtreme_bt2111(result, x, y, width, height, standard);
    } else {
        pixtreme_full_field(result, x, width, standard);
    }
}

extern "C" __global__ void pixtreme_color_bars_f32(
    float* __restrict__ output,
    const long long width,
    const long long height,
    const int standard
) {
    const long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long y = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    int code[3];
    pixtreme_color_bar_code(code, x, y, width, height, standard);
    const long long offset = (y * width + x) * 3LL;
    if (standard == 4) {
        output[offset] = (float)code[0] / 1023.0f;
        output[offset + 1] = (float)code[1] / 1023.0f;
        output[offset + 2] = (float)code[2] / 1023.0f;
    } else {
        output[offset] = ((float)code[0] - 64.0f) / 876.0f;
        output[offset + 1] = ((float)code[1] - 64.0f) / 876.0f;
        output[offset + 2] = ((float)code[2] - 64.0f) / 876.0f;
    }
}

extern "C" __global__ void pixtreme_color_bars_u16(
    unsigned short* __restrict__ output,
    const long long width,
    const long long height,
    const int standard
) {
    const long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long y = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    int code[3];
    pixtreme_color_bar_code(code, x, y, width, height, standard);
    const long long offset = (y * width + x) * 3LL;
    output[offset] = (unsigned short)code[0];
    output[offset + 1] = (unsigned short)code[1];
    output[offset + 2] = (unsigned short)code[2];
}
"""


@lru_cache(maxsize=1)
def _geometry_kernel() -> cp.RawKernel:
    return cp.RawKernel(_GEOMETRY_KERNEL_SOURCE, "pixtreme_generate_geometry")


@lru_cache(maxsize=2)
def _color_bars_kernel(output: str) -> cp.RawKernel:
    name = "pixtreme_color_bars_f32" if output == "normalized" else "pixtreme_color_bars_u16"
    return cp.RawKernel(_COLOR_BARS_KERNEL_SOURCE, name)


def _dimension(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            _actionable_error(
                why=f"{name} must be a positive integer pixel count",
                what=f"received {name}={value!r}",
                how=f"pass {name} as an int greater than 0",
            )
        )
    return value


def _host_array(value: object) -> np.ndarray:
    return _validation._host_array(value, why=_HOST_ARRAY_WHY, how=_HOST_ARRAY_HOW)


def _color(value: object, *, name: str) -> tuple[float, ...]:
    try:
        array = _host_array(value)
    except ValueError:
        array = np.asarray((), dtype=np.float32)
    if array.ndim != 1 or array.shape[0] not in (1, 3, 4):
        raise ValueError(
            _actionable_error(
                why=f"{name} must be a real color sequence of length 1, 3, or 4",
                what=f"received {name} shape {array.shape!r}",
                how=f"pass {name} as one, three, or four finite real values",
            )
        )
    return tuple(
        _finite_real(item.item() if isinstance(item, np.generic) else item, name=f"{name}[{index}]")
        for index, item in enumerate(array)
    )


def _matching_colors(
    first: object,
    second: object,
    *,
    first_name: str,
    second_name: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    first_color = _color(first, name=first_name)
    second_color = _color(second, name=second_name)
    if len(first_color) != len(second_color):
        raise ValueError(
            _actionable_error(
                why="all color arguments in one generator call must have the same channel count",
                what=f"received {first_name} length {len(first_color)} and {second_name} length {len(second_color)}",
                how="pass both colors with the same supported length: 1, 3, or 4",
            )
        )
    return first_color, second_color


def _checker_colors(value: object) -> tuple[tuple[float, ...], tuple[float, ...]]:
    try:
        sequence = tuple(cast(Sequence[object], value))
    except TypeError as error:
        raise ValueError(
            _actionable_error(
                why="colors must contain exactly two color sequences",
                what=f"received colors={value!r}",
                how="pass colors=(first_color, second_color)",
            )
        ) from error
    if len(sequence) != 2:
        raise ValueError(
            _actionable_error(
                why="colors must contain exactly two color sequences",
                what=f"received {len(sequence)} colors",
                how="pass colors=(first_color, second_color)",
            )
        )
    return _matching_colors(sequence[0], sequence[1], first_name="colors[0]", second_name="colors[1]")


def _metadata(colorspace: object, gamma: object) -> tuple[str, str]:
    return (
        _normalized_closed_token(colorspace, axis="colorspace", accepted=_COLORSPACE_TOKENS),
        _normalized_closed_token(gamma, axis="gamma", accepted=_GAMMA_TOKENS),
    )


def _channel_labels(channel_count: int) -> tuple[str, ...]:
    return {1: ("Y",), 3: ("R", "G", "B"), 4: ("R", "G", "B", "A")}[channel_count]


@lru_cache(maxsize=256)
def _device_color(device_id: int, color: tuple[float, ...]) -> cp.ndarray:
    del device_id
    return cp.asarray(color, dtype=cp.float32)


def _generate_geometry(
    *,
    width: int,
    height: int,
    first_color: tuple[float, ...],
    second_color: tuple[float, ...],
    geometry: int,
    aa: str,
    parameters: tuple[float, float, float, float, float],
    colorspace: str,
    gamma: str,
) -> Frame:
    output = cp.empty((height, width, len(first_color)), dtype=cp.float32)
    grid = (
        (width + _GENERATOR_BLOCK[0] - 1) // _GENERATOR_BLOCK[0],
        (height + _GENERATOR_BLOCK[1] - 1) // _GENERATOR_BLOCK[1],
    )
    device_id = cp.cuda.runtime.getDevice()
    _geometry_kernel()(
        grid,
        _GENERATOR_BLOCK,
        (
            output,
            _device_color(device_id, first_color),
            _device_color(device_id, second_color),
            np.int64(width),
            np.int64(height),
            np.int64(len(first_color)),
            np.int32(geometry),
            np.int32(_AA_TOKENS.index(aa)),
            *(np.float32(value) for value in parameters),
        ),
    )
    return Frame(
        data=output,
        colorspace=colorspace,
        gamma=gamma,
        channels=_channel_labels(len(first_color)),
    )


def ramp(
    *,
    width: int,
    height: int,
    kind: str = "linear",
    start: Sequence[float],
    end: Sequence[float],
    start_color: Sequence[float],
    end_color: Sequence[float],
    colorspace: str,
    gamma: str = "linear",
) -> Frame:
    """Generate a two-color ramp in continuous pixel coordinates.

    Coordinates are ``(x, y)`` and pixel ``(i, j)`` is evaluated at
    ``(i + 0.5, j + 0.5)``. Linear projection or radial distance saturates to
    the endpoint colors, then mixes fp32 values directly in the declared gamma
    space. Scene values are not clamped. The returned Frame has colorspace and
    gamma from this call, channels derived from color length, and a new
    C-contiguous allocation.
    """
    checked_width = _dimension(width, name="width")
    checked_height = _dimension(height, name="height")
    checked_kind = _normalized_closed_token(kind, axis="kind", accepted=_KIND_TOKENS)
    checked_start = _finite_pair(start, name="start")
    checked_end = _finite_pair(end, name="end")
    if checked_start == checked_end:
        raise ValueError(
            _actionable_error(
                why="start and end must define a non-zero ramp direction or radius",
                what=f"received start=end={checked_start!r}",
                how="pass distinct (x, y) points for start and end",
            )
        )
    checked_start_color, checked_end_color = _matching_colors(
        start_color,
        end_color,
        first_name="start_color",
        second_name="end_color",
    )
    checked_colorspace, checked_gamma = _metadata(colorspace, gamma)
    return _generate_geometry(
        width=checked_width,
        height=checked_height,
        first_color=checked_start_color,
        second_color=checked_end_color,
        geometry=_GEOMETRY_RAMP_LINEAR if checked_kind == "linear" else _GEOMETRY_RAMP_RADIAL,
        aa="off",
        parameters=(*checked_start, *checked_end, 0.0),
        colorspace=checked_colorspace,
        gamma=checked_gamma,
    )


def grid(
    *,
    width: int,
    height: int,
    cell: float | Sequence[float],
    line_width: float,
    color: Sequence[float],
    background: Sequence[float],
    offset: Sequence[float] = (0.0, 0.0),
    colorspace: str,
    gamma: str = "linear",
    aa: str = "distance",
) -> Frame:
    """Generate a periodic grid in continuous pixel coordinates.

    Coordinates are ``(x, y)`` with pixel centers at ``(i + 0.5, j + 0.5)``.
    Lines are centered on the offset lattice, and crossing coverage is united
    before one color mix. AA is distance, fixed 4x4 supersample, or binary off.
    Scene values are not clamped. Metadata uses the requested colorspace,
    gamma, and color-derived channels; output owns a new C-contiguous allocation.
    """
    checked_width = _dimension(width, name="width")
    checked_height = _dimension(height, name="height")
    checked_cell = _positive_scalar_or_pair(cell, name="cell", why=_CELL_WHY, how=_CELL_HOW)
    checked_line_width = _positive_real(line_width, name="line_width")
    checked_color, checked_background = _matching_colors(
        color,
        background,
        first_name="color",
        second_name="background",
    )
    checked_offset = _finite_pair(offset, name="offset")
    checked_colorspace, checked_gamma = _metadata(colorspace, gamma)
    checked_aa = _normalized_closed_token(aa, axis="aa", accepted=_AA_TOKENS)
    return _generate_geometry(
        width=checked_width,
        height=checked_height,
        first_color=checked_color,
        second_color=checked_background,
        geometry=_GEOMETRY_GRID,
        aa=checked_aa,
        parameters=(*checked_cell, checked_line_width, *checked_offset),
        colorspace=checked_colorspace,
        gamma=checked_gamma,
    )


def checkerboard(
    *,
    width: int,
    height: int,
    cell: float | Sequence[float],
    colors: Sequence[Sequence[float]],
    offset: Sequence[float] = (0.0, 0.0),
    colorspace: str,
    gamma: str = "linear",
    aa: str = "distance",
) -> Frame:
    """Generate a periodic two-color checkerboard in continuous coordinates.

    The first color occupies the origin cell. Coordinates are ``(x, y)`` and
    pixel ``(i, j)`` is sampled at ``(i + 0.5, j + 0.5)`` with distance,
    fixed 4x4 supersample, or binary-off AA. Mixing occurs in the declared
    gamma space without scene-value clamp. Metadata uses the requested
    colorspace, gamma, and color-derived channels; output owns a new allocation.
    """
    checked_width = _dimension(width, name="width")
    checked_height = _dimension(height, name="height")
    checked_cell = _positive_scalar_or_pair(cell, name="cell", why=_CELL_WHY, how=_CELL_HOW)
    first_color, second_color = _checker_colors(colors)
    checked_offset = _finite_pair(offset, name="offset")
    checked_colorspace, checked_gamma = _metadata(colorspace, gamma)
    checked_aa = _normalized_closed_token(aa, axis="aa", accepted=_AA_TOKENS)
    return _generate_geometry(
        width=checked_width,
        height=checked_height,
        first_color=first_color,
        second_color=second_color,
        geometry=_GEOMETRY_CHECKERBOARD,
        aa=checked_aa,
        parameters=(*checked_cell, *checked_offset, 0.0),
        colorspace=checked_colorspace,
        gamma=checked_gamma,
    )


def color_bars(
    *,
    width: int,
    height: int,
    standard: str,
    output: str = "normalized",
) -> Frame:
    """Generate one standards-bound RGB color-bar Frame.

    ``standard`` determines integer-aligned region geometry, colorspace,
    gamma, and RGB channels. ``normalized`` returns fp32 full-range values:
    narrow standards use ``(code - 64) / 876`` and PQ full range uses
    ``code / 1023``. ``code`` returns the exact 10-bit values in a uint16
    Frame; scale management remains the caller's responsibility. Bars have no
    edge AA, scene sub-black remains negative after normalization, and every
    call returns a new C-contiguous allocation.
    """
    checked_width = _dimension(width, name="width")
    checked_height = _dimension(height, name="height")
    checked_standard = _normalized_closed_token(standard, axis="standard", accepted=_STANDARD_TOKENS)
    checked_output = _normalized_closed_token(output, axis="output", accepted=_OUTPUT_TOKENS)
    standard_index = _STANDARD_TOKENS.index(checked_standard)
    dtype = cp.float32 if checked_output == "normalized" else cp.uint16
    data = cp.empty((checked_height, checked_width, 3), dtype=dtype)
    grid = (
        (checked_width + _GENERATOR_BLOCK[0] - 1) // _GENERATOR_BLOCK[0],
        (checked_height + _GENERATOR_BLOCK[1] - 1) // _GENERATOR_BLOCK[1],
    )
    _color_bars_kernel(checked_output)(
        grid,
        _GENERATOR_BLOCK,
        (
            data,
            np.int64(checked_width),
            np.int64(checked_height),
            np.int32(standard_index),
        ),
    )
    colorspace = "Rec.2020" if checked_standard.startswith("bt2111") else "Rec.709"
    if checked_standard == "bt2111-hlg":
        gamma = "hlg"
    elif checked_standard.startswith("bt2111-pq"):
        gamma = "pq"
    else:
        gamma = "rec709"
    return Frame(data=data, colorspace=colorspace, gamma=gamma, channels=("R", "G", "B"))
