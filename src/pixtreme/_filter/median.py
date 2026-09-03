"""GPU median blur."""

from __future__ import annotations

from functools import lru_cache

import cupy as cp
import numpy as np

from pixtreme._core.border import _BORDER_PREAMBLE, _border_argument, _resolve_border
from pixtreme._core.frame import Frame, _new_frame, _validate_float32_frame
from pixtreme._core.vocabulary import Border
from pixtreme._filter.common import (
    _RAW_KERNEL_BLOCK,
    _RAW_KERNEL_SHARED_LIMIT,
    _shape_arguments,
    _validate_odd_size,
)

# Per-size JIT probes keep size 7 at 200 local bytes/thread; size 9 crosses
# the chosen 256-byte bound at 328 bytes/thread and raises insertion-sort work.
_MEDIAN_MAX_SIZE = 7


_MEDIAN_ROW_SORT_NETWORKS = {
    1: (),
    3: ((0, 1), (1, 2), (0, 1)),
    5: ((0, 3), (1, 4), (0, 2), (1, 3), (0, 1), (2, 4), (1, 2), (3, 4), (2, 3)),
}


_MEDIAN_FALLBACK_OPERATION_TEMPLATE = r"""
const long long channel = i % channel_count;
const long long pixel = i / channel_count;
const long long x = pixel % width;
const long long y = pixel / width;
const long long radius = __SIZE__ / 2;
float values[__COUNT__];
int sample = 0;

for (long long offset_y = -radius; offset_y <= radius; ++offset_y) {
    for (long long offset_x = -radius; offset_x <= radius; ++offset_x) {
        values[sample] = pixtreme_border_sample(
            source, x + offset_x, y + offset_y, width, height, channel_count, channel, border, border_value
        );
        ++sample;
    }
}

for (int current = 1; current < __COUNT__; ++current) {
    const float key = values[current];
    int previous = current - 1;
    while (previous >= 0 && values[previous] > key) {
        values[previous + 1] = values[previous];
        --previous;
    }
    values[previous + 1] = key;
}

filtered = values[__MEDIAN__];
"""


@lru_cache(maxsize=4)
def _median_fallback_kernel(size: int) -> cp.ElementwiseKernel:
    count = size * size
    operation = (
        _MEDIAN_FALLBACK_OPERATION_TEMPLATE.replace("__SIZE__", str(size))
        .replace("__COUNT__", str(count))
        .replace("__MEDIAN__", str(count // 2))
    )
    return cp.ElementwiseKernel(
        "raw T source, int64 width, int64 height, int64 channel_count, int32 border, float32 border_value",
        "float32 filtered",
        operation,
        f"pixtreme_blur_median_fallback_{size}",
        preamble=_BORDER_PREAMBLE,
    )


def _median_kernel_source(size: int) -> str:
    radius = size // 2
    rows = [[f"value_{row}_{column}" for column in range(size)] for row in range(size)]
    lines = [
        _BORDER_PREAMBLE,
        f"""
extern "C" __global__ void pixtreme_blur_median_{size}(
    const float* __restrict__ source,
    float* __restrict__ output,
    const long long width,
    const long long height,
    const long long channel_count,
    const int border,
    const float border_value
) {{
    extern __shared__ float tile[];
    const long long radius = {radius};
    const long long halo = radius * channel_count;
    const long long tile_width = blockDim.x + 2 * halo;
    const long long tile_height = blockDim.y + 2 * radius;
    const long long thread_index = threadIdx.y * blockDim.x + threadIdx.x;
    const long long thread_count = blockDim.x * blockDim.y;
    const long long tile_count = tile_width * tile_height;
    const long long block_xc = (long long)blockIdx.x * blockDim.x;
    const long long block_y = (long long)blockIdx.y * blockDim.y;

    for (long long tile_index = thread_index; tile_index < tile_count; tile_index += thread_count) {{
        const long long local_y = tile_index / tile_width;
        const long long local_xc = tile_index - local_y * tile_width;
        const long long source_xc = block_xc + local_xc - halo;
        const long long channel = pixtreme_positive_modulo(source_xc, channel_count);
        const long long source_x = (source_xc - channel) / channel_count;
        tile[tile_index] = pixtreme_border_sample(
            source,
            source_x,
            block_y + local_y - radius,
            width,
            height,
            channel_count,
            channel,
            border,
            border_value
        );
    }}
    __syncthreads();

    const long long output_xc = block_xc + threadIdx.x;
    const long long output_y = block_y + threadIdx.y;
    const long long row_elements = width * channel_count;
    if (output_xc >= row_elements || output_y >= height) {{
        return;
    }}
""",
    ]
    if size == _MEDIAN_MAX_SIZE:

        def tile_value(sample: int) -> str:
            row, column = divmod(sample, size)
            return f"tile[(threadIdx.y + {row}) * tile_width + threadIdx.x + {column} * channel_count]"

        def compare_swap(left: str, right: str) -> str:
            return (
                f"    {{ const float low = fminf({left}, {right}); {right} = fmaxf({left}, {right}); {left} = low; }}\n"
            )

        active = [f"candidate_{index}" for index in range(size * size // 2 + 2)]
        for sample, name in enumerate(active):
            lines.append(f"    float {name} = {tile_value(sample)};\n")
        next_sample = len(active)
        while len(active) > 3:
            pair_count = len(active) // 2
            for pair in range(pair_count):
                lines.append(compare_swap(active[2 * pair], active[2 * pair + 1]))
            for pair in range(1, pair_count):
                lines.append(compare_swap(active[0], active[2 * pair]))
            if len(active) % 2:
                lines.append(compare_swap(active[0], active[-1]))
            maximum_target = active[-1]
            maximum_positions = range(1, 2 * pair_count, 2)
            if len(active) % 2 == 0:
                maximum_positions = range(1, 2 * pair_count - 1, 2)
            for position in maximum_positions:
                lines.append(compare_swap(active[position], maximum_target))
            lines.append(f"    {active[0]} = {tile_value(next_sample)};\n")
            next_sample += 1
            active.pop()
        lines.append(compare_swap(active[0], active[1]))
        lines.append(compare_swap(active[1], active[2]))
        lines.append(compare_swap(active[0], active[1]))
        lines.append(f"    output[output_y * row_elements + output_xc] = {active[1]};\n")
        lines.append("}\n")
        return "".join(lines)
    for row, names in enumerate(rows):
        for column, name in enumerate(names):
            lines.append(
                f"    float {name} = tile[(threadIdx.y + {row}) * tile_width + "
                f"threadIdx.x + {column} * channel_count];\n"
            )
    for names in rows:
        for left_index, right_index in _MEDIAN_ROW_SORT_NETWORKS[size]:
            left = names[left_index]
            right = names[right_index]
            lines.append(
                f"    if ({left} > {right}) {{ const float swap = {left}; {left} = {right}; {right} = swap; }}\n"
            )
    median_index = size * size // 2
    for selection in range(median_index + 1):
        lines.append(f"    float selected_{selection} = {rows[0][0]};\n")
        lines.append(f"    int selected_row_{selection} = 0;\n")
        for row in range(1, size):
            lines.append(
                f"    if ({rows[row][0]} < selected_{selection}) "
                f"{{ selected_{selection} = {rows[row][0]}; selected_row_{selection} = {row}; }}\n"
            )
        if selection == median_index:
            continue
        for row, names in enumerate(rows):
            lines.append(f"    if (selected_row_{selection} == {row}) {{\n")
            for column in range(size - 1):
                lines.append(f"        {names[column]} = {names[column + 1]};\n")
            lines.append(f"        {names[-1]} = __int_as_float(0x7f800000);\n")
            lines.append("    }\n")
    lines.append(f"    output[output_y * row_elements + output_xc] = selected_{median_index};\n")
    lines.append("}\n")
    return "".join(lines)


@lru_cache(maxsize=4)
def _median_kernel(size: int) -> cp.RawKernel:
    return cp.RawKernel(
        _median_kernel_source(size),
        f"pixtreme_blur_median_{size}",
    )


def median_blur(
    frame: Frame,
    *,
    size: int,
    border: Border = "mirror",
    border_value: float | None = None,
) -> Frame:
    """Replace each channel value with its square-window median.

    ``size`` is a positive odd integer from 1 through 7. Border defaults to
    ``mirror`` (edge-excluding reflection); ``replicate`` clamps to the edge and
    ``wrap`` uses periodic indices. ``constant`` uses ``border_value`` for every
    virtual pixel outside the image; ``border_value`` is required with
    ``constant`` and forbidden for every other border. Median selection is
    independent per channel, uses fp32 values, and does not clamp negative
    values or values above 1. The result always owns new storage, including
    size 1.
    """
    checked_frame = _validate_float32_frame(frame, operation="filter.median_blur")
    checked_size = _validate_odd_size(size, operation="filter.median_blur", maximum=_MEDIAN_MAX_SIZE)
    checked_border, checked_border_value = _resolve_border(border, border_value)
    output = cp.empty(checked_frame.shape, dtype=cp.float32)
    block_x, block_y = _RAW_KERNEL_BLOCK
    row_elements = checked_frame.width * len(checked_frame.channels)
    radius = checked_size // 2
    grid = ((row_elements + block_x - 1) // block_x, (checked_frame.height + block_y - 1) // block_y)
    shared_elements = (block_x + 2 * radius * len(checked_frame.channels)) * (block_y + 2 * radius)
    shared_bytes = shared_elements * np.dtype(np.float32).itemsize
    if shared_bytes <= _RAW_KERNEL_SHARED_LIMIT:
        _median_kernel(checked_size)(
            grid,
            _RAW_KERNEL_BLOCK,
            (
                checked_frame.data,
                output,
                *_shape_arguments(checked_frame),
                _border_argument(checked_border),
                np.float32(checked_border_value),
            ),
            shared_mem=shared_bytes,
        )
    else:
        _median_fallback_kernel(checked_size)(
            checked_frame.data,
            *_shape_arguments(checked_frame),
            _border_argument(checked_border),
            np.float32(checked_border_value),
            output,
        )
    return _new_frame(checked_frame, output)
