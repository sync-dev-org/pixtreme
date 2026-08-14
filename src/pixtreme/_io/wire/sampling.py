"""Shared H.273 range resolution and chroma sampling helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, cast

import cupy as cp
import numpy as np

from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import _COLORSPACE_TOKENS, _GAMMA_TOKENS, _MATRIX_TOKENS, Frame
from pixtreme._core.interpolation import _POINT_INTERPOLATION_TOKENS, _specialized_point_weight_source
from pixtreme._core.validation import _normalized_closed_token
from pixtreme._core.value_domain import _float32_conversion_guidance
from pixtreme._core.vocabulary import _CHROMA_SITING_TOKENS

if TYPE_CHECKING:
    pass

_INTERPOLATION_TOKENS = _POINT_INTERPOLATION_TOKENS
_SITING_TOKENS = _CHROMA_SITING_TOKENS
_SITING_OFFSETS = {
    "left": (0.0, 0.5),
    "center": (0.5, 0.5),
    "topleft": (0.0, 0.0),
}
_PLANAR_BIT_DEPTHS = {
    "from_yuv420p": (8, 10),
    "from_yuv422p": (8, 10, 12),
    "from_yuv444p": (10, 12),
    "from_yuva444p": (12,),
    "to_yuv420p": (8, 10),
    "to_yuv422p": (8, 10, 12),
    "to_yuv444p": (10, 12),
    "to_yuva444p": (12,),
}

_PLACEHOLDER_COLORSPACE = "Rec.709"
_PLACEHOLDER_GAMMA = "rec709"
_YCBCR_CHANNELS = ("Y", "Cb", "Cr")
_YCBCRA_CHANNELS = ("Y", "Cb", "Cr", "A")
_THREADS_PER_BLOCK = 256
_token = _normalized_closed_token

_SUBSAMPLED_KERNEL_TEMPLATE = r"""
typedef __INPUT_TYPE__ pixtreme_input_t;

__WEIGHT_FUNCTION__

__READ_FUNCTIONS__

extern "C" __global__ void pixtreme_from_subsampled(
    const pixtreme_input_t* __restrict__ input,
    float* __restrict__ output,
    const long long pixel_count,
    const int width,
    const int height,
    const int chroma_width,
    const int chroma_height,
    const int row_words,
    const float y_offset,
    const float y_scale,
    const float chroma_offset,
    const float chroma_scale
) {
    const long long pixel = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (pixel >= pixel_count) {
        return;
    }
    const int x = (int)(pixel % width);
    const int y = (int)(pixel / width);
    const unsigned int y_code = pixtreme_read_y(input, pixel, x, y, width, row_words);

    __SAMPLE_CHROMA__

    const long long output_base = pixel * 3LL;
    output[output_base] = ((float)y_code - y_offset) * y_scale;
    output[output_base + 1] = (cb_code - chroma_offset) * chroma_scale;
    output[output_base + 2] = (cr_code - chroma_offset) * chroma_scale;
}
"""

_PLANAR_444_KERNEL_TEMPLATE = r"""
typedef __INPUT_TYPE__ pixtreme_input_t;

extern "C" __global__ void pixtreme_from_planar_444(
    const pixtreme_input_t* __restrict__ input,
    float* __restrict__ output,
    const long long pixel_count,
    const float y_offset,
    const float y_scale,
    const float chroma_offset,
    const float chroma_scale,
    const float alpha_scale
) {
    const long long pixel = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (pixel >= pixel_count) {
        return;
    }
    const unsigned int y_code = ((unsigned int)input[pixel]) & __CODE_MASK__;
    const unsigned int cb_code = ((unsigned int)input[pixel_count + pixel]) & __CODE_MASK__;
    const unsigned int cr_code = ((unsigned int)input[2LL * pixel_count + pixel]) & __CODE_MASK__;
    const long long output_base = pixel * __CHANNEL_COUNT__LL;
    output[output_base] = ((float)y_code - y_offset) * y_scale;
    output[output_base + 1] = ((float)cb_code - chroma_offset) * chroma_scale;
    output[output_base + 2] = ((float)cr_code - chroma_offset) * chroma_scale;
    __ALPHA_STORE__
}
"""


def _input_type(bit_depth: int, *, v210: bool = False) -> str:
    if v210:
        return "unsigned int"
    return "unsigned char" if bit_depth == 8 else "unsigned short"


def _float_literal(value: float) -> str:
    return f"{value:.1f}f"


def _tap_geometry(interpolation: str) -> tuple[int, int]:
    if interpolation == "bilinear":
        return 0, 2
    if interpolation.startswith("lanczos"):
        lobes = int(interpolation.removeprefix("lanczos"))
        return -(lobes - 1), 2 * lobes
    return -1, 4


def _sample_chroma(interpolation: str, *, vertical_subsampling: bool, offset: tuple[float, float]) -> str:
    source_x = f"((float)x - {_float_literal(offset[0])}) * 0.5f"
    source_y = f"((float)y - {_float_literal(offset[1])}) * 0.5f"
    if interpolation == "nearest":
        sample_y = f"pixtreme_clamp((int)floorf({source_y} + 0.5f), chroma_height)" if vertical_subsampling else "y"
        return f"""
    const int sample_x = pixtreme_clamp((int)floorf({source_x} + 0.5f), chroma_width);
    const int sample_y = {sample_y};
    const float cb_code = (float)pixtreme_read_cb(
        input, sample_x, sample_y, width, chroma_width, chroma_height, pixel_count, row_words
    );
    const float cr_code = (float)pixtreme_read_cr(
        input, sample_x, sample_y, width, chroma_width, chroma_height, pixel_count, row_words
    );
"""

    start_offset, tap_count = _tap_geometry(interpolation)
    if vertical_subsampling:
        vertical_setup = f"""
    const float source_y = {source_y};
    const int start_y = (int)floorf(source_y) + ({start_offset});
"""
        vertical_loop = f"""
    for (int tap_y = 0; tap_y < {tap_count}; ++tap_y) {{
        const int raw_y = start_y + tap_y;
        const int sample_y = pixtreme_clamp(raw_y, chroma_height);
        const float weight_y = pixtreme_weight(source_y - (float)raw_y);
"""
    else:
        vertical_setup = ""
        vertical_loop = """
    for (int tap_y = 0; tap_y < 1; ++tap_y) {
        const int sample_y = y;
        const float weight_y = 1.0f;
"""
    return f"""
    const float source_x = {source_x};
    const int start_x = (int)floorf(source_x) + ({start_offset});
{vertical_setup}
    float cb_code = 0.0f;
    float cr_code = 0.0f;
    float weight_sum = 0.0f;
{vertical_loop}
        for (int tap_x = 0; tap_x < {tap_count}; ++tap_x) {{
            const int raw_x = start_x + tap_x;
            const int sample_x = pixtreme_clamp(raw_x, chroma_width);
            const float weight_x = pixtreme_weight(source_x - (float)raw_x);
            const float weight = weight_x * weight_y;
            cb_code += (float)pixtreme_read_cb(
                input, sample_x, sample_y, width, chroma_width, chroma_height, pixel_count, row_words
            ) * weight;
            cr_code += (float)pixtreme_read_cr(
                input, sample_x, sample_y, width, chroma_width, chroma_height, pixel_count, row_words
            ) * weight;
            weight_sum += weight;
        }}
    }}
    cb_code = weight_sum != 0.0f ? cb_code / weight_sum : 0.0f;
    cr_code = weight_sum != 0.0f ? cr_code / weight_sum : 0.0f;
"""


def _read_functions(layout: str, *, bit_depth: int) -> str:
    mask = (1 << bit_depth) - 1
    common = """
__device__ int pixtreme_clamp(const int index, const int extent) {
    return index < 0 ? 0 : (index >= extent ? extent - 1 : index);
}
"""
    if layout == "uyvy422":
        return (
            common
            + r"""
__device__ unsigned int pixtreme_read_y(
    const pixtreme_input_t* input,
    const long long pixel,
    const int x,
    const int y,
    const int width,
    const int row_words
) {
    (void)x;
    (void)y;
    (void)width;
    (void)row_words;
    const long long pair_base = (pixel >> 1) * 4LL;
    return (unsigned int)input[pair_base + ((pixel & 1LL) ? 3LL : 1LL)];
}

__device__ unsigned int pixtreme_read_cb(
    const pixtreme_input_t* input,
    const int sample_x,
    const int sample_y,
    const int width,
    const int chroma_width,
    const int chroma_height,
    const long long pixel_count,
    const int row_words
) {
    (void)chroma_width;
    (void)chroma_height;
    (void)pixel_count;
    (void)row_words;
    return (unsigned int)input[(long long)sample_y * width * 2LL + sample_x * 4LL];
}

__device__ unsigned int pixtreme_read_cr(
    const pixtreme_input_t* input,
    const int sample_x,
    const int sample_y,
    const int width,
    const int chroma_width,
    const int chroma_height,
    const long long pixel_count,
    const int row_words
) {
    (void)chroma_width;
    (void)chroma_height;
    (void)pixel_count;
    (void)row_words;
    return (unsigned int)input[(long long)sample_y * width * 2LL + sample_x * 4LL + 2LL];
}
"""
        )
    if layout == "v210":
        return (
            common
            + r"""
__device__ unsigned int pixtreme_read_y(
    const pixtreme_input_t* input,
    const long long pixel,
    const int x,
    const int y,
    const int width,
    const int row_words
) {
    (void)pixel;
    (void)width;
    const int group = x / 6;
    const int position = x - group * 6;
    const long long base = (long long)y * row_words + group * 4LL;
    if (position == 0) {
        return (input[base] >> 10) & 1023U;
    }
    if (position == 1) {
        return input[base + 1] & 1023U;
    }
    if (position == 2) {
        return (input[base + 1] >> 20) & 1023U;
    }
    if (position == 3) {
        return (input[base + 2] >> 10) & 1023U;
    }
    if (position == 4) {
        return input[base + 3] & 1023U;
    }
    return (input[base + 3] >> 20) & 1023U;
}

__device__ unsigned int pixtreme_read_cb(
    const pixtreme_input_t* input,
    const int sample_x,
    const int sample_y,
    const int width,
    const int chroma_width,
    const int chroma_height,
    const long long pixel_count,
    const int row_words
) {
    (void)width;
    (void)chroma_width;
    (void)chroma_height;
    (void)pixel_count;
    const int group = sample_x / 3;
    const int position = sample_x - group * 3;
    const long long base = (long long)sample_y * row_words + group * 4LL;
    if (position == 0) {
        return input[base] & 1023U;
    }
    if (position == 1) {
        return (input[base + 1] >> 10) & 1023U;
    }
    return (input[base + 2] >> 20) & 1023U;
}

__device__ unsigned int pixtreme_read_cr(
    const pixtreme_input_t* input,
    const int sample_x,
    const int sample_y,
    const int width,
    const int chroma_width,
    const int chroma_height,
    const long long pixel_count,
    const int row_words
) {
    (void)width;
    (void)chroma_width;
    (void)chroma_height;
    (void)pixel_count;
    const int group = sample_x / 3;
    const int position = sample_x - group * 3;
    const long long base = (long long)sample_y * row_words + group * 4LL;
    if (position == 0) {
        return (input[base] >> 20) & 1023U;
    }
    if (position == 1) {
        return input[base + 2] & 1023U;
    }
    return (input[base + 3] >> 10) & 1023U;
}
"""
        )
    if layout in {"yuv420p", "yuv422p"}:
        return (
            common
            + f"""
__device__ unsigned int pixtreme_read_y(
    const pixtreme_input_t* input,
    const long long pixel,
    const int x,
    const int y,
    const int width,
    const int row_words
) {{
    (void)x;
    (void)y;
    (void)width;
    (void)row_words;
    return ((unsigned int)input[pixel]) & {mask}U;
}}

__device__ unsigned int pixtreme_read_cb(
    const pixtreme_input_t* input,
    const int sample_x,
    const int sample_y,
    const int width,
    const int chroma_width,
    const int chroma_height,
    const long long pixel_count,
    const int row_words
) {{
    (void)width;
    (void)chroma_height;
    (void)row_words;
    const long long chroma_index = (long long)sample_y * chroma_width + sample_x;
    return ((unsigned int)input[pixel_count + chroma_index]) & {mask}U;
}}

__device__ unsigned int pixtreme_read_cr(
    const pixtreme_input_t* input,
    const int sample_x,
    const int sample_y,
    const int width,
    const int chroma_width,
    const int chroma_height,
    const long long pixel_count,
    const int row_words
) {{
    (void)width;
    (void)row_words;
    const long long chroma_size = (long long)chroma_width * chroma_height;
    const long long chroma_index = (long long)sample_y * chroma_width + sample_x;
    return ((unsigned int)input[pixel_count + chroma_size + chroma_index]) & {mask}U;
}}
"""
        )
    shift = 6 if layout == "p010" else 0
    return (
        common
        + f"""
__device__ unsigned int pixtreme_read_y(
    const pixtreme_input_t* input,
    const long long pixel,
    const int x,
    const int y,
    const int width,
    const int row_words
) {{
    (void)x;
    (void)y;
    (void)width;
    (void)row_words;
    return (((unsigned int)input[pixel]) >> {shift}) & {mask}U;
}}

__device__ unsigned int pixtreme_read_cb(
    const pixtreme_input_t* input,
    const int sample_x,
    const int sample_y,
    const int width,
    const int chroma_width,
    const int chroma_height,
    const long long pixel_count,
    const int row_words
) {{
    (void)width;
    (void)chroma_height;
    (void)row_words;
    const long long chroma_index = (long long)sample_y * chroma_width + sample_x;
    return (((unsigned int)input[pixel_count + chroma_index * 2LL]) >> {shift}) & {mask}U;
}}

__device__ unsigned int pixtreme_read_cr(
    const pixtreme_input_t* input,
    const int sample_x,
    const int sample_y,
    const int width,
    const int chroma_width,
    const int chroma_height,
    const long long pixel_count,
    const int row_words
) {{
    (void)width;
    (void)chroma_height;
    (void)row_words;
    const long long chroma_index = (long long)sample_y * chroma_width + sample_x;
    return (((unsigned int)input[pixel_count + chroma_index * 2LL + 1LL]) >> {shift}) & {mask}U;
}}
"""
    )


@lru_cache(maxsize=None)
def _subsampled_kernel_source(
    layout: str,
    bit_depth: int,
    interpolation: str,
    siting: str,
) -> str:
    vertical_subsampling = layout in {"yuv420p", "nv12", "p010"}
    offset = _SITING_OFFSETS[siting] if vertical_subsampling else (0.0, 0.0)
    source = (
        _SUBSAMPLED_KERNEL_TEMPLATE.replace(
            "__INPUT_TYPE__",
            _input_type(bit_depth, v210=layout == "v210"),
        )
        .replace("__WEIGHT_FUNCTION__", _specialized_point_weight_source(interpolation))
        .replace("__READ_FUNCTIONS__", _read_functions(layout, bit_depth=bit_depth))
        .replace(
            "__SAMPLE_CHROMA__",
            _sample_chroma(
                interpolation,
                vertical_subsampling=vertical_subsampling,
                offset=offset,
            ),
        )
    )
    return source


@lru_cache(maxsize=None)
def _planar_444_kernel_source(bit_depth: int, *, alpha: bool) -> str:
    source = (
        _PLANAR_444_KERNEL_TEMPLATE.replace("__INPUT_TYPE__", _input_type(bit_depth))
        .replace("__CODE_MASK__", str((1 << bit_depth) - 1))
        .replace("__CHANNEL_COUNT__", "4" if alpha else "3")
        .replace(
            "__ALPHA_STORE__",
            (
                "const unsigned int alpha_code = ((unsigned int)input[3LL * pixel_count + pixel]) & "
                f"{(1 << bit_depth) - 1}U;\n"
                "    output[output_base + 3] = (float)alpha_code * alpha_scale;"
                if alpha
                else "(void)alpha_scale;"
            ),
        )
    )
    return source


def _bit_depth(value: object, *, operation: str) -> int:
    accepted = _PLANAR_BIT_DEPTHS[operation]
    if type(value) is not int or value not in accepted:
        raise ValueError(
            _actionable_error(
                why=f"{operation} bit_depth is a closed integer domain",
                what=f"received bit_depth={value!r}",
                how=f"pass one of {accepted!r}",
            )
        )
    return value


def _dimension(value: object, *, name: str, even: bool) -> int:
    valid = type(value) is int and value > 0 and (not even or value % 2 == 0)
    if not valid:
        requirement = "a positive even integer" if even else "a positive integer"
        raise ValueError(
            _actionable_error(
                why=f"{name} must be {requirement}",
                what=f"received {name}={value!r}",
                how=f"pass {name} as {requirement}",
            )
        )
    return cast(int, value)


def _dimensions(
    width: object,
    height: object,
    *,
    even_width: bool,
    even_height: bool,
) -> tuple[int, int]:
    return (
        _dimension(width, name="width", even=even_width),
        _dimension(height, name="height", even=even_height),
    )


def _validate_buffer(
    buf: object,
    *,
    operation: str,
    dtype: np.dtype[np.generic],
    element_count: int,
    shapes: tuple[tuple[int, ...], ...],
) -> cp.ndarray:
    if not isinstance(buf, cp.ndarray):
        raise ValueError(
            _actionable_error(
                why=f"{operation} consumes a CUDA-resident cupy.ndarray buffer",
                what=f"received {type(buf).__module__}.{type(buf).__qualname__}",
                how=f"transfer or wrap the buffer as cupy.ndarray with dtype {dtype.name}",
            )
        )
    if np.dtype(buf.dtype) != dtype:
        raise ValueError(
            _actionable_error(
                why=f"{operation} container dtype is fixed by its packing contract",
                what=f"received buffer dtype {buf.dtype!s}",
                how=f"pass a {dtype.name} cupy.ndarray",
            )
        )
    if buf.size != element_count:
        raise ValueError(
            _actionable_error(
                why=f"{operation} buffer size must exactly match width, height, and packing",
                what=f"received {buf.size} elements",
                how=f"pass exactly {element_count} elements",
            )
        )
    if tuple(buf.shape) not in shapes:
        raise ValueError(
            _actionable_error(
                why=f"{operation} accepts only its documented buffer shapes",
                what=f"received shape {buf.shape!r}",
                how=f"reshape without copying to one of {shapes!r}",
            )
        )
    if not buf.flags.c_contiguous:
        raise ValueError(
            _actionable_error(
                why=f"{operation} layout is defined over one C-contiguous buffer",
                what=f"received strides {buf.strides!r}",
                how="pass cp.ascontiguousarray(buf) before decoding",
            )
        )
    return buf.reshape(-1)


def _range_parameters(value: str, *, bit_depth: int) -> tuple[np.float32, np.float32, np.float32, np.float32]:
    if value == "full":
        scale = np.float32(1.0 / ((1 << bit_depth) - 1))
        return np.float32(0.0), scale, np.float32(0.0), scale
    code_scale = 1 << (bit_depth - 8)
    return (
        np.float32(16 * code_scale),
        np.float32(1.0 / (219 * code_scale)),
        np.float32(16 * code_scale),
        np.float32(1.0 / (224 * code_scale)),
    )


def _launch_shape(pixel_count: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return ((pixel_count + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK,), (_THREADS_PER_BLOCK,)


def _metadata(colorspace: str | None, gamma: str | None) -> tuple[str, str]:
    return (
        _PLACEHOLDER_COLORSPACE
        if colorspace is None
        else _token(colorspace, axis="colorspace", accepted=_COLORSPACE_TOKENS),
        _PLACEHOLDER_GAMMA if gamma is None else _token(gamma, axis="gamma", accepted=_GAMMA_TOKENS),
    )


def _matrix(value: str | None) -> str | None:
    return None if value is None else _token(value, axis="matrix", accepted=_MATRIX_TOKENS)


def _frame(
    output: cp.ndarray,
    *,
    colorspace: str,
    gamma: str,
    matrix: str | None,
    alpha: bool = False,
) -> Frame:
    return Frame(
        data=output,
        colorspace=colorspace,
        gamma=gamma,
        channels=_YCBCRA_CHANNELS if alpha else _YCBCR_CHANNELS,
        matrix=matrix,
    )


def _from_subsampled(
    input_data: cp.ndarray,
    kernel: cp.RawKernel,
    *,
    layout: str,
    width: int,
    height: int,
    bit_depth: int,
    range: str,
    colorspace: str,
    gamma: str,
    matrix: str | None,
    row_words: int = 0,
) -> Frame:
    pixel_count = width * height
    chroma_width = (width + 1) // 2 if layout == "v210" else width // 2
    chroma_height = height // 2 if layout in {"yuv420p", "nv12", "p010"} else height
    output = cp.empty((height, width, 3), dtype=cp.float32)
    grid, block = _launch_shape(pixel_count)
    kernel(
        grid,
        block,
        (
            input_data,
            output,
            np.int64(pixel_count),
            np.int32(width),
            np.int32(height),
            np.int32(chroma_width),
            np.int32(chroma_height),
            np.int32(row_words),
            *_range_parameters(range, bit_depth=bit_depth),
        ),
    )
    return _frame(output, colorspace=colorspace, gamma=gamma, matrix=matrix)


def _from_planar_444(
    input_data: cp.ndarray,
    kernel: cp.RawKernel,
    *,
    width: int,
    height: int,
    bit_depth: int,
    range: str,
    alpha: bool,
    colorspace: str,
    gamma: str,
    matrix: str | None,
) -> Frame:
    pixel_count = width * height
    channel_count = 4 if alpha else 3
    output = cp.empty((height, width, channel_count), dtype=cp.float32)
    grid, block = _launch_shape(pixel_count)
    kernel(
        grid,
        block,
        (
            input_data,
            output,
            np.int64(pixel_count),
            *_range_parameters(range, bit_depth=bit_depth),
            np.float32(1.0 / ((1 << bit_depth) - 1)),
        ),
    )
    return _frame(output, colorspace=colorspace, gamma=gamma, matrix=matrix, alpha=alpha)


_TO_INTERPOLATION_TOKENS = (*_POINT_INTERPOLATION_TOKENS[:3], "area")

_TO_SUBSAMPLED_KERNEL_TEMPLATE = r"""
typedef __OUTPUT_TYPE__ pixtreme_output_t;

__WEIGHT_FUNCTION__

__SAMPLE_FUNCTION__

__STORE_HELPERS__

extern "C" __global__ void pixtreme_to_subsampled(
    const float* __restrict__ input,
    pixtreme_output_t* __restrict__ output,
    const long long work_count,
    const long long pixel_count,
    const int width,
    const int height,
    const int chroma_width,
    const int chroma_height,
    const int row_words,
    const float y_offset,
    const float y_scale,
    const float chroma_offset,
    const float chroma_scale,
    const unsigned int maximum
) {
    const long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= work_count) {
        return;
    }

    __STORE_BODY__
}
"""

_TO_PLANAR_444_KERNEL_TEMPLATE = r"""
typedef unsigned short pixtreme_output_t;

__QUANTIZE_FUNCTION__

extern "C" __global__ void pixtreme_to_planar_444(
    const float* __restrict__ input,
    pixtreme_output_t* __restrict__ output,
    const long long pixel_count,
    const float y_offset,
    const float y_scale,
    const float chroma_offset,
    const float chroma_scale,
    const unsigned int maximum
) {
    const long long pixel = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (pixel >= pixel_count) {
        return;
    }
    const long long input_base = pixel * __CHANNEL_COUNT__LL;
    output[pixel] = (pixtreme_output_t)pixtreme_quantize(
        input[input_base], y_offset, y_scale, maximum
    );
    output[pixel_count + pixel] = (pixtreme_output_t)pixtreme_quantize(
        input[input_base + 1], chroma_offset, chroma_scale, maximum
    );
    output[2LL * pixel_count + pixel] = (pixtreme_output_t)pixtreme_quantize(
        input[input_base + 2], chroma_offset, chroma_scale, maximum
    );
    __ALPHA_STORE__
}
"""


def _output_type(bit_depth: int, *, v210: bool = False) -> str:
    if v210:
        return "unsigned int"
    return "unsigned char" if bit_depth == 8 else "unsigned short"


def _to_float_literal(value: float) -> str:
    return f"{value:.1f}f"


def _to_weight_function(interpolation: str) -> str:
    if interpolation == "area":
        body = """
    const float x = fabsf(distance);
    if (x <= 0.5f) {
        return 1.0f;
    }
    if (x < 1.5f) {
        return 1.5f - x;
    }
    return 0.0f;
"""
        return f"""
__device__ float pixtreme_weight(const float distance) {{
{body}
}}
"""
    return _specialized_point_weight_source(interpolation, distance_scale="0.5f")


def _to_tap_geometry(interpolation: str) -> tuple[int, int]:
    if interpolation == "area":
        return -2, 6
    if interpolation == "bilinear":
        return -1, 4
    return -3, 8


def _sample_function(
    interpolation: str,
    *,
    vertical_subsampling: bool,
    offset: tuple[float, float],
) -> str:
    center_x = f"(float)output_x * 2.0f + {_to_float_literal(offset[0])}"
    center_y = f"(float)output_y * 2.0f + {_to_float_literal(offset[1])}" if vertical_subsampling else "(float)output_y"
    common = """
__device__ int pixtreme_clamp(const int index, const int extent) {
    return index < 0 ? 0 : (index >= extent ? extent - 1 : index);
}
"""
    if interpolation == "nearest":
        return (
            common
            + f"""
__device__ float pixtreme_sample_chroma(
    const float* input,
    const int output_x,
    const int output_y,
    const int channel,
    const int width,
    const int height
) {{
    const float center_x = {center_x};
    const float center_y = {center_y};
    const int sample_x = pixtreme_clamp((int)floorf(center_x + 0.5f), width);
    const int sample_y = pixtreme_clamp((int)floorf(center_y + 0.5f), height);
    return input[((long long)sample_y * width + sample_x) * 3LL + channel];
}}
"""
        )

    start_offset, tap_count = _to_tap_geometry(interpolation)
    if vertical_subsampling:
        vertical_setup = f"""
    const int base_y = (int)floorf(center_y);
    const int start_y = base_y + ({start_offset});
"""
        vertical_loop = f"""
    for (int tap_y = 0; tap_y < {tap_count}; ++tap_y) {{
        const int raw_y = start_y + tap_y;
        const int sample_y = pixtreme_clamp(raw_y, height);
        const float weight_y = pixtreme_weight((float)raw_y - center_y);
"""
    else:
        vertical_setup = ""
        vertical_loop = """
    for (int tap_y = 0; tap_y < 1; ++tap_y) {
        const int sample_y = output_y;
        const float weight_y = 1.0f;
"""
    return (
        common
        + f"""
__device__ float pixtreme_sample_chroma(
    const float* input,
    const int output_x,
    const int output_y,
    const int channel,
    const int width,
    const int height
) {{
    const float center_x = {center_x};
    const float center_y = {center_y};
    const int base_x = (int)floorf(center_x);
    const int start_x = base_x + ({start_offset});
{vertical_setup}
    float value_sum = 0.0f;
    float weight_sum = 0.0f;
{vertical_loop}
        for (int tap_x = 0; tap_x < {tap_count}; ++tap_x) {{
            const int raw_x = start_x + tap_x;
            const int sample_x = pixtreme_clamp(raw_x, width);
            const float weight_x = pixtreme_weight((float)raw_x - center_x);
            const float weight = weight_x * weight_y;
            value_sum += input[((long long)sample_y * width + sample_x) * 3LL + channel] * weight;
            weight_sum += weight;
        }}
    }}
    return weight_sum != 0.0f ? value_sum / weight_sum : 0.0f;
}}
"""
    )


def _quantize_function() -> str:
    return r"""
__device__ unsigned int pixtreme_quantize(
    const float value,
    const float offset,
    const float scale,
    const unsigned int maximum
) {
    const float mapped = value * scale + offset;
    const float rounded = mapped >= 0.0f ? floorf(mapped + 0.5f) : ceilf(mapped - 0.5f);
    if (rounded <= 0.0f) {
        return 0U;
    }
    if (rounded >= (float)maximum) {
        return maximum;
    }
    return (unsigned int)rounded;
}
"""


def _store_helpers() -> str:
    return (
        _quantize_function()
        + r"""
__device__ unsigned int pixtreme_y_code(
    const float* input,
    const int x,
    const int y,
    const int width,
    const float offset,
    const float scale,
    const unsigned int maximum
) {
    const int sample_x = pixtreme_clamp(x, width);
    return pixtreme_quantize(
        input[((long long)y * width + sample_x) * 3LL],
        offset,
        scale,
        maximum
    );
}

__device__ unsigned int pixtreme_chroma_code(
    const float* input,
    const int output_x,
    const int output_y,
    const int channel,
    const int width,
    const int height,
    const float offset,
    const float scale,
    const unsigned int maximum
) {
    return pixtreme_quantize(
        pixtreme_sample_chroma(input, output_x, output_y, channel, width, height),
        offset,
        scale,
        maximum
    );
}
"""
    )


def _store_body(layout: str) -> str:
    if layout in {"yuv420p", "yuv422p"}:
        return r"""
    if (index < pixel_count) {
        output[index] = (pixtreme_output_t)pixtreme_quantize(
            input[index * 3LL], y_offset, y_scale, maximum
        );
    }
    const long long chroma_count = (long long)chroma_width * chroma_height;
    if (index < chroma_count) {
        const int output_x = (int)(index % chroma_width);
        const int output_y = (int)(index / chroma_width);
        output[pixel_count + index] = (pixtreme_output_t)pixtreme_chroma_code(
            input, output_x, output_y, 1, width, height,
            chroma_offset, chroma_scale, maximum
        );
        output[pixel_count + chroma_count + index] = (pixtreme_output_t)pixtreme_chroma_code(
            input, output_x, output_y, 2, width, height,
            chroma_offset, chroma_scale, maximum
        );
    }
"""
    if layout in {"nv12", "p010"}:
        shift = " << 6" if layout == "p010" else ""
        return f"""
    if (index < pixel_count) {{
        output[index] = (pixtreme_output_t)(
            pixtreme_quantize(input[index * 3LL], y_offset, y_scale, maximum){shift}
        );
    }}
    const long long chroma_count = (long long)chroma_width * chroma_height;
    if (index < chroma_count) {{
        const int output_x = (int)(index % chroma_width);
        const int output_y = (int)(index / chroma_width);
        output[pixel_count + index * 2LL] = (pixtreme_output_t)(
            pixtreme_chroma_code(
                input, output_x, output_y, 1, width, height,
                chroma_offset, chroma_scale, maximum
            ){shift}
        );
        output[pixel_count + index * 2LL + 1LL] = (pixtreme_output_t)(
            pixtreme_chroma_code(
                input, output_x, output_y, 2, width, height,
                chroma_offset, chroma_scale, maximum
            ){shift}
        );
    }}
"""
    if layout == "uyvy422":
        return r"""
    const int output_x = (int)(index % chroma_width);
    const int output_y = (int)(index / chroma_width);
    const int input_x = output_x * 2;
    const long long output_base = index * 4LL;
    output[output_base] = (pixtreme_output_t)pixtreme_chroma_code(
        input, output_x, output_y, 1, width, height,
        chroma_offset, chroma_scale, maximum
    );
    output[output_base + 1] = (pixtreme_output_t)pixtreme_y_code(
        input, input_x, output_y, width, y_offset, y_scale, maximum
    );
    output[output_base + 2] = (pixtreme_output_t)pixtreme_chroma_code(
        input, output_x, output_y, 2, width, height,
        chroma_offset, chroma_scale, maximum
    );
    output[output_base + 3] = (pixtreme_output_t)pixtreme_y_code(
        input, input_x + 1, output_y, width, y_offset, y_scale, maximum
    );
"""
    return r"""
    const int output_y = (int)(index / row_words);
    const int word_x = (int)(index - (long long)output_y * row_words);
    const int active_words = ((width + 5) / 6) * 4;
    if (word_x >= active_words) {
        output[index] = 0U;
        return;
    }
    const int group = word_x / 4;
    const int position = word_x - group * 4;
    const int input_x = group * 6;
    const int chroma_x = group * 3;
    unsigned int low;
    unsigned int middle;
    unsigned int high;
    if (position == 0) {
        low = pixtreme_chroma_code(
            input, pixtreme_clamp(chroma_x, chroma_width), output_y, 1, width, height,
            chroma_offset, chroma_scale, maximum
        );
        middle = pixtreme_y_code(
            input, input_x, output_y, width, y_offset, y_scale, maximum
        );
        high = pixtreme_chroma_code(
            input, pixtreme_clamp(chroma_x, chroma_width), output_y, 2, width, height,
            chroma_offset, chroma_scale, maximum
        );
    } else if (position == 1) {
        low = pixtreme_y_code(
            input, input_x + 1, output_y, width, y_offset, y_scale, maximum
        );
        middle = pixtreme_chroma_code(
            input, pixtreme_clamp(chroma_x + 1, chroma_width), output_y, 1, width, height,
            chroma_offset, chroma_scale, maximum
        );
        high = pixtreme_y_code(
            input, input_x + 2, output_y, width, y_offset, y_scale, maximum
        );
    } else if (position == 2) {
        low = pixtreme_chroma_code(
            input, pixtreme_clamp(chroma_x + 1, chroma_width), output_y, 2, width, height,
            chroma_offset, chroma_scale, maximum
        );
        middle = pixtreme_y_code(
            input, input_x + 3, output_y, width, y_offset, y_scale, maximum
        );
        high = pixtreme_chroma_code(
            input, pixtreme_clamp(chroma_x + 2, chroma_width), output_y, 1, width, height,
            chroma_offset, chroma_scale, maximum
        );
    } else {
        low = pixtreme_y_code(
            input, input_x + 4, output_y, width, y_offset, y_scale, maximum
        );
        middle = pixtreme_chroma_code(
            input, pixtreme_clamp(chroma_x + 2, chroma_width), output_y, 2, width, height,
            chroma_offset, chroma_scale, maximum
        );
        high = pixtreme_y_code(
            input, input_x + 5, output_y, width, y_offset, y_scale, maximum
        );
    }
    output[index] = low | (middle << 10) | (high << 20);
"""


@lru_cache(maxsize=None)
def _to_subsampled_kernel_source(
    layout: str,
    bit_depth: int,
    interpolation: str,
    siting: str,
) -> str:
    vertical_subsampling = layout in {"yuv420p", "nv12", "p010"}
    offset = _SITING_OFFSETS[siting] if vertical_subsampling else (0.0, 0.0)
    source = (
        _TO_SUBSAMPLED_KERNEL_TEMPLATE.replace(
            "__OUTPUT_TYPE__",
            _output_type(bit_depth, v210=layout == "v210"),
        )
        .replace("__WEIGHT_FUNCTION__", _to_weight_function(interpolation))
        .replace(
            "__SAMPLE_FUNCTION__",
            _sample_function(
                interpolation,
                vertical_subsampling=vertical_subsampling,
                offset=offset,
            ),
        )
        .replace("__STORE_HELPERS__", _store_helpers())
        .replace("__STORE_BODY__", _store_body(layout))
    )
    return source


@lru_cache(maxsize=None)
def _to_planar_444_kernel_source(*, alpha: bool) -> str:
    source = (
        _TO_PLANAR_444_KERNEL_TEMPLATE.replace("__QUANTIZE_FUNCTION__", _quantize_function())
        .replace("__CHANNEL_COUNT__", "4" if alpha else "3")
        .replace(
            "__ALPHA_STORE__",
            (
                "output[3LL * pixel_count + pixel] = (pixtreme_output_t)pixtreme_quantize(\n"
                "        input[input_base + 3], 0.0f, (float)maximum, maximum\n"
                "    );"
                if alpha
                else ""
            ),
        )
    )
    return source


def _validate_frame(frame: Frame, *, operation: str, alpha: bool = False) -> None:
    expected_channels = _YCBCRA_CHANNELS if alpha else _YCBCR_CHANNELS
    if frame.channels != expected_channels:
        raise ValueError(
            _actionable_error(
                why=f"{operation} requires exact label-driven {expected_channels!r} channels",
                what=f"received Frame channels {frame.channels!r}",
                how=(
                    "use px.color.rgb_to_ycbcr(frame) before export"
                    if not alpha
                    else "use px.color.rgb_to_ycbcr(frame) to produce Y, Cb, Cr, A channels before export"
                ),
            )
        )
    if np.dtype(frame.dtype) != np.dtype(np.float32):
        raise ValueError(
            _actionable_error(
                why=f"{operation} quantizes from fp32 working values",
                what=f"received Frame dtype {frame.dtype!s}",
                how=_float32_conversion_guidance(np.dtype(frame.dtype)),
            )
        )


def _to_range_parameters(value: str, *, bit_depth: int) -> tuple[np.float32, np.float32, np.float32, np.float32]:
    if value == "full":
        maximum = np.float32((1 << bit_depth) - 1)
        return np.float32(0.0), maximum, np.float32(0.0), maximum
    code_scale = 1 << (bit_depth - 8)
    return (
        np.float32(16 * code_scale),
        np.float32(219 * code_scale),
        np.float32(16 * code_scale),
        np.float32(224 * code_scale),
    )


def _to_subsampled(
    frame: Frame,
    kernel: cp.RawKernel,
    *,
    layout: str,
    bit_depth: int,
    range: str,
) -> cp.ndarray:
    width, height = frame.width, frame.height
    pixel_count = width * height
    chroma_width = (width + 1) // 2 if layout == "v210" else width // 2
    chroma_height = height // 2 if layout in {"yuv420p", "nv12", "p010"} else height
    row_words = ((width + 47) // 48) * 32 if layout == "v210" else 0
    if layout == "uyvy422":
        element_count = pixel_count * 2
        work_count = pixel_count // 2
        dtype = cp.uint8
    elif layout == "v210":
        element_count = row_words * height
        work_count = element_count
        dtype = cp.uint32
    elif layout in {"nv12", "p010"}:
        element_count = pixel_count + pixel_count // 2
        work_count = pixel_count
        dtype = cp.uint16 if layout == "p010" else cp.uint8
    else:
        chroma_count = chroma_width * chroma_height
        element_count = pixel_count + 2 * chroma_count
        work_count = pixel_count
        dtype = cp.uint8 if bit_depth == 8 else cp.uint16
    output = cp.empty((element_count,), dtype=dtype)
    grid, block = _launch_shape(work_count)
    kernel(
        grid,
        block,
        (
            frame.data,
            output,
            np.int64(work_count),
            np.int64(pixel_count),
            np.int32(width),
            np.int32(height),
            np.int32(chroma_width),
            np.int32(chroma_height),
            np.int32(row_words),
            *_to_range_parameters(range, bit_depth=bit_depth),
            np.uint32((1 << bit_depth) - 1),
        ),
    )
    return output


def _to_planar_444(
    frame: Frame,
    kernel: cp.RawKernel,
    *,
    bit_depth: int,
    range: str,
    alpha: bool,
) -> cp.ndarray:
    pixel_count = frame.width * frame.height
    channel_count = 4 if alpha else 3
    output = cp.empty((pixel_count * channel_count,), dtype=cp.uint16)
    grid, block = _launch_shape(pixel_count)
    kernel(
        grid,
        block,
        (
            frame.data,
            output,
            np.int64(pixel_count),
            *_to_range_parameters(range, bit_depth=bit_depth),
            np.uint32((1 << bit_depth) - 1),
        ),
    )
    return output
