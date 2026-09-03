"""GPU morphology filters with explicit structuring-element and border contracts."""

from __future__ import annotations

import math
from functools import lru_cache

import cupy as cp
import numpy as np

from pixtreme._core.border import _BORDER_PREAMBLE, _border_argument, _resolve_border
from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame, _new_frame, _validate_frame
from pixtreme._core.validation import _normalized_closed_token
from pixtreme._core.value_domain import _float32_conversion_guidance
from pixtreme._core.vocabulary import _MORPHOLOGY_SHAPE_TOKENS, Border, MorphologyShape

_SHAPE_TOKENS = _MORPHOLOGY_SHAPE_TOKENS
_THREADS_PER_BLOCK = 256
_MORPHOLOGY_BLOCK = (32, 16)
_MORPHOLOGY_SHARED_LIMIT = 48 * 1024
_MORPHOLOGY_TILED_MAX_CHANNELS = 4
_MORPHOLOGY_ROW_LIMIT_CACHE_SIZE = 32
_DIFFERENCE_NONE = 0
_DIFFERENCE_SOURCE_MINUS_RESULT = 1
_DIFFERENCE_RESULT_MINUS_SOURCE = 2

_MORPHOLOGY_KERNEL_SOURCE = (
    _BORDER_PREAMBLE
    + r"""
extern "C" __global__ void pixtreme_morphology(
    const float* __restrict__ source,
    float* __restrict__ output,
    const long long width,
    const long long height,
    const long long channel_count,
    const long long radius,
    const int shape,
    const int border,
    const float border_value,
    const int dilate
) {
    const long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    const long long element_count = width * height * channel_count;
    if (index >= element_count) {
        return;
    }

    const long long channel = index % channel_count;
    const long long pixel_index = index / channel_count;
    const long long x = pixel_index % width;
    const long long y = pixel_index / width;
    float value = source[index];
    const double radius_squared = (double)radius * (double)radius;

    for (long long offset_y = -radius; offset_y <= radius; ++offset_y) {
        for (long long offset_x = -radius; offset_x <= radius; ++offset_x) {
            if (
                shape == 0
                && (double)offset_x * (double)offset_x + (double)offset_y * (double)offset_y > radius_squared
            ) {
                continue;
            }
            const float sample = pixtreme_border_sample(
                source,
                x + offset_x,
                y + offset_y,
                width,
                height,
                channel_count,
                channel,
                border,
                border_value
            );
            value = dilate ? fmaxf(value, sample) : fminf(value, sample);
        }
    }
    output[index] = value;
}
"""
)


@lru_cache(maxsize=1)
def _morphology_kernel() -> cp.RawKernel:
    return cp.RawKernel(_MORPHOLOGY_KERNEL_SOURCE, "pixtreme_morphology")


def _morphology_tiled_kernel_source(channel_count: int, shape: str, operation: str) -> str:
    kernel_name = f"pixtreme_morphology_tiled_{channel_count}_{shape}_{operation}"
    disk_parameter = "    const int* __restrict__ disk_row_limits,\n" if shape == "disk" else ""
    horizontal_limit = "disk_row_limits[offset_y + radius]" if shape == "disk" else "radius"
    lines = [
        _BORDER_PREAMBLE,
        f"""
extern "C" __global__ void {kernel_name}(
    const float* __restrict__ source,
    float* __restrict__ output,
    const float* __restrict__ difference_source,
{disk_parameter}    const long long width,
    const long long height,
    const int radius,
    const int border,
    const float border_value,
    const int difference_mode
) {{
    extern __shared__ float tile[];
    const int channel_count = {channel_count};
    const int tile_width = blockDim.x + 2 * radius;
    const int tile_height = blockDim.y + 2 * radius;
    const int tile_pixel_count = tile_width * tile_height;
    const int tile_count = tile_pixel_count * channel_count;
    const int thread_index = threadIdx.y * blockDim.x + threadIdx.x;
    const int thread_count = blockDim.x * blockDim.y;
    const long long block_x = (long long)blockIdx.x * blockDim.x;
    const long long block_y = (long long)blockIdx.y * blockDim.y;
    const bool tile_is_interior =
        block_x >= radius && block_y >= radius
        && block_x + blockDim.x + radius <= width
        && block_y + blockDim.y + radius <= height;

    for (int tile_index = thread_index; tile_index < tile_count; tile_index += thread_count) {{
        const int channel = tile_index % channel_count;
        const int tile_pixel = tile_index / channel_count;
        const int local_x = tile_pixel % tile_width;
        const int local_y = tile_pixel / tile_width;
        const long long source_x = block_x + local_x - radius;
        const long long source_y = block_y + local_y - radius;
        if (tile_is_interior) {{
            tile[tile_index] = source[(source_y * width + source_x) * channel_count + channel];
        }} else {{
            tile[tile_index] = pixtreme_border_sample(
                source,
                source_x,
                source_y,
                width,
                height,
                channel_count,
                channel,
                border,
                border_value
            );
        }}
    }}
    __syncthreads();

    const long long output_x = block_x + threadIdx.x;
    const long long output_y = block_y + threadIdx.y;
    if (output_x >= width || output_y >= height) {{
        return;
    }}
    const int center_index =
        ((threadIdx.y + radius) * tile_width + threadIdx.x + radius) * channel_count;
""",
    ]
    if operation == "gradient":
        for channel in range(channel_count):
            lines.append(f"    float minimum_{channel} = tile[center_index + {channel}];\n")
            lines.append(f"    float maximum_{channel} = tile[center_index + {channel}];\n")
    else:
        for channel in range(channel_count):
            lines.append(f"    float value_{channel} = tile[center_index + {channel}];\n")
    lines.extend(
        (
            "    for (int offset_y = -radius; offset_y <= radius; ++offset_y) {\n",
            f"        const int horizontal_limit = {horizontal_limit};\n",
            "        for (int offset_x = -horizontal_limit; offset_x <= horizontal_limit; ++offset_x) {\n",
            "            const int neighbor_index =\n",
            "                ((threadIdx.y + radius + offset_y) * tile_width + threadIdx.x + radius + offset_x)\n",
            "                * channel_count;\n",
        )
    )
    if operation == "gradient":
        for channel in range(channel_count):
            lines.append(
                f"            minimum_{channel} = fminf(minimum_{channel}, tile[neighbor_index + {channel}]);\n"
            )
            lines.append(
                f"            maximum_{channel} = fmaxf(maximum_{channel}, tile[neighbor_index + {channel}]);\n"
            )
    else:
        extremum = "fmaxf" if operation == "dilate" else "fminf"
        for channel in range(channel_count):
            lines.append(
                f"            value_{channel} = {extremum}(value_{channel}, tile[neighbor_index + {channel}]);\n"
            )
    lines.extend(
        (
            "        }\n",
            "    }\n",
            "    const long long output_index = (output_y * width + output_x) * channel_count;\n",
        )
    )
    for channel in range(channel_count):
        result = f"maximum_{channel} - minimum_{channel}" if operation == "gradient" else f"value_{channel}"
        lines.append(f"    float result_{channel} = {result};\n")
        lines.append(
            f"    if (difference_mode == 1) result_{channel} = difference_source[output_index + {channel}] - result_{channel};\n"
        )
        lines.append(
            f"    if (difference_mode == 2) result_{channel} = result_{channel} - difference_source[output_index + {channel}];\n"
        )
        lines.append(f"    output[output_index + {channel}] = result_{channel};\n")
    lines.append("}\n")
    return "".join(lines)


@lru_cache(maxsize=6 * _MORPHOLOGY_TILED_MAX_CHANNELS)
def _morphology_tiled_kernel(channel_count: int, shape: str, operation: str) -> cp.RawKernel:
    return cp.RawKernel(
        _morphology_tiled_kernel_source(channel_count, shape, operation),
        f"pixtreme_morphology_tiled_{channel_count}_{shape}_{operation}",
    )


@lru_cache(maxsize=_MORPHOLOGY_ROW_LIMIT_CACHE_SIZE)
def _disk_row_limits(device_id: int, radius: int) -> cp.ndarray:
    limits = np.asarray(
        [math.isqrt(radius * radius - offset_y * offset_y) for offset_y in range(-radius, radius + 1)],
        dtype=np.int32,
    )
    with cp.cuda.Device(device_id):
        return cp.asarray(limits)


def _validate_radius(radius: object, *, operation: str) -> int:
    if type(radius) is not int or radius < 1:
        raise ValueError(
            _actionable_error(
                why=f"{operation} radius must be a built-in int of at least 1",
                what=f"received radius={radius!r}",
                how="pass radius as an int greater than or equal to 1",
            )
        )
    return radius


def _validate_shape(shape: object) -> MorphologyShape:
    return _normalized_closed_token(shape, axis="shape", accepted=_SHAPE_TOKENS)


def _validate_arguments(
    frame: object,
    *,
    operation: str,
    radius: object,
    shape: object,
    border: object,
    border_value: object,
) -> tuple[Frame, int, MorphologyShape, Border, float]:
    checked_frame = _validate_frame(frame, operation=operation)
    dtype = np.dtype(checked_frame.dtype)
    if dtype != np.dtype(np.float32):
        raise ValueError(
            _actionable_error(
                why=f"{operation} requires float32 Frame data",
                what=f"received Frame data dtype {dtype.name}",
                how=_float32_conversion_guidance(dtype),
            )
        )
    checked_radius = _validate_radius(radius, operation=operation)
    checked_shape = _validate_shape(shape)
    checked_border, checked_border_value = _resolve_border(border, border_value)
    return checked_frame, checked_radius, checked_shape, checked_border, checked_border_value


def _launch_tiled(
    frame: Frame,
    output: cp.ndarray,
    *,
    radius: int,
    shape: str,
    border: Border,
    border_value: float,
    operation: str,
    difference_source: cp.ndarray,
    difference_mode: int,
) -> bool:
    channel_count = len(frame.channels)
    block_x, block_y = _MORPHOLOGY_BLOCK
    shared_elements = (block_x + 2 * radius) * (block_y + 2 * radius) * channel_count
    shared_bytes = shared_elements * np.dtype(np.float32).itemsize
    if channel_count > _MORPHOLOGY_TILED_MAX_CHANNELS or shared_bytes > _MORPHOLOGY_SHARED_LIMIT:
        return False
    grid = ((frame.width + block_x - 1) // block_x, (frame.height + block_y - 1) // block_y)
    kernel_arguments: tuple[object, ...] = (frame.data, output, difference_source)
    if shape == "disk":
        kernel_arguments += (_disk_row_limits(int(cp.cuda.runtime.getDevice()), radius),)
    kernel_arguments += (
        np.int64(frame.width),
        np.int64(frame.height),
        np.int32(radius),
        _border_argument(border),
        np.float32(border_value),
        np.int32(difference_mode),
    )
    _morphology_tiled_kernel(channel_count, shape, operation)(
        grid,
        _MORPHOLOGY_BLOCK,
        kernel_arguments,
        shared_mem=shared_bytes,
    )
    return True


def _primitive(
    frame: Frame,
    *,
    radius: int,
    shape: str,
    border: Border,
    border_value: float,
    dilate: bool,
    difference_source: cp.ndarray | None = None,
    difference_mode: int = _DIFFERENCE_NONE,
) -> Frame:
    output = cp.empty(frame.shape, dtype=cp.float32)
    resolved_difference_source = frame.data if difference_source is None else difference_source
    operation = "dilate" if dilate else "erode"
    if not _launch_tiled(
        frame,
        output,
        radius=radius,
        shape=shape,
        border=border,
        border_value=border_value,
        operation=operation,
        difference_source=resolved_difference_source,
        difference_mode=difference_mode,
    ):
        channel_count = len(frame.channels)
        element_count = int(frame.data.size)
        blocks = (element_count + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
        _morphology_kernel()(
            (blocks,),
            (_THREADS_PER_BLOCK,),
            (
                frame.data,
                output,
                np.int64(frame.width),
                np.int64(frame.height),
                np.int64(channel_count),
                np.int64(radius),
                np.int32(_SHAPE_TOKENS.index(shape)),
                _border_argument(border),
                np.float32(border_value),
                np.int32(dilate),
            ),
        )
        if difference_mode == _DIFFERENCE_SOURCE_MINUS_RESULT:
            output = resolved_difference_source - output
        elif difference_mode == _DIFFERENCE_RESULT_MINUS_SOURCE:
            output = output - resolved_difference_source
    return _new_frame(frame, output)


def _gradient_primitive(
    frame: Frame,
    *,
    radius: int,
    shape: str,
    border: Border,
    border_value: float,
) -> Frame:
    output = cp.empty(frame.shape, dtype=cp.float32)
    if _launch_tiled(
        frame,
        output,
        radius=radius,
        shape=shape,
        border=border,
        border_value=border_value,
        operation="gradient",
        difference_source=frame.data,
        difference_mode=_DIFFERENCE_NONE,
    ):
        return _new_frame(frame, output)
    eroded = _primitive(
        frame,
        radius=radius,
        shape=shape,
        border=border,
        border_value=border_value,
        dilate=False,
    )
    dilated = _primitive(
        frame,
        radius=radius,
        shape=shape,
        border=border,
        border_value=border_value,
        dilate=True,
    )
    return _new_frame(frame, dilated.data - eroded.data)


def _validated_primitive(
    frame: Frame,
    *,
    operation: str,
    radius: int,
    shape: str,
    border: Border,
    border_value: float | None,
    dilate: bool,
) -> Frame:
    checked_frame, checked_radius, checked_shape, checked_border, checked_border_value = _validate_arguments(
        frame,
        operation=operation,
        radius=radius,
        shape=shape,
        border=border,
        border_value=border_value,
    )
    return _primitive(
        checked_frame,
        radius=checked_radius,
        shape=checked_shape,
        border=checked_border,
        border_value=checked_border_value,
        dilate=dilate,
    )


def erosion(
    frame: Frame,
    *,
    radius: int,
    shape: MorphologyShape = "disk",
    border: Border = "replicate",
    border_value: float | None = None,
) -> Frame:
    """Replace every fp32 channel value with its structuring-element minimum.

    ``radius`` is an int of at least 1. ``disk`` includes integer offsets whose
    squared distance is at most ``radius ** 2``; ``square`` includes the full
    ``(2 * radius + 1)`` window. Border defaults to neutral ``replicate``;
    ``mirror`` reflects without repeating the edge, ``wrap`` is periodic, and
    ``constant`` requires a finite ``border_value``. Processing is independent
    per channel, preserves Frame metadata, and does not clamp scene values.
    """
    return _validated_primitive(
        frame,
        operation="morphology.erosion",
        radius=radius,
        shape=shape,
        border=border,
        border_value=border_value,
        dilate=False,
    )


def dilation(
    frame: Frame,
    *,
    radius: int,
    shape: MorphologyShape = "disk",
    border: Border = "replicate",
    border_value: float | None = None,
) -> Frame:
    """Replace every fp32 channel value with its structuring-element maximum.

    ``radius`` is an int of at least 1. ``disk`` includes integer offsets whose
    squared distance is at most ``radius ** 2``; ``square`` includes the full
    ``(2 * radius + 1)`` window. Border defaults to neutral ``replicate``;
    ``mirror`` reflects without repeating the edge, ``wrap`` is periodic, and
    ``constant`` requires a finite ``border_value``. Processing is independent
    per channel, preserves Frame metadata, and does not clamp scene values.
    """
    return _validated_primitive(
        frame,
        operation="morphology.dilation",
        radius=radius,
        shape=shape,
        border=border,
        border_value=border_value,
        dilate=True,
    )


def opening(
    frame: Frame,
    *,
    radius: int,
    shape: MorphologyShape = "disk",
    border: Border = "replicate",
    border_value: float | None = None,
) -> Frame:
    """Erode then dilate an fp32 Frame with one shared structuring element.

    ``radius`` is an int of at least 1; ``shape`` is ``disk`` or ``square``.
    Both stages use the same border, defaulting to neutral ``replicate``;
    ``constant`` requires finite ``border_value``. The per-channel operation
    preserves metadata and does not clamp negative values or values above 1.
    """
    checked_frame, checked_radius, checked_shape, checked_border, checked_border_value = _validate_arguments(
        frame,
        operation="morphology.opening",
        radius=radius,
        shape=shape,
        border=border,
        border_value=border_value,
    )
    eroded = _primitive(
        checked_frame,
        radius=checked_radius,
        shape=checked_shape,
        border=checked_border,
        border_value=checked_border_value,
        dilate=False,
    )
    return _primitive(
        eroded,
        radius=checked_radius,
        shape=checked_shape,
        border=checked_border,
        border_value=checked_border_value,
        dilate=True,
    )


def closing(
    frame: Frame,
    *,
    radius: int,
    shape: MorphologyShape = "disk",
    border: Border = "replicate",
    border_value: float | None = None,
) -> Frame:
    """Dilate then erode an fp32 Frame with one shared structuring element.

    ``radius`` is an int of at least 1; ``shape`` is ``disk`` or ``square``.
    Both stages use the same border, defaulting to neutral ``replicate``;
    ``constant`` requires finite ``border_value``. The per-channel operation
    preserves metadata and does not clamp negative values or values above 1.
    """
    checked_frame, checked_radius, checked_shape, checked_border, checked_border_value = _validate_arguments(
        frame,
        operation="morphology.closing",
        radius=radius,
        shape=shape,
        border=border,
        border_value=border_value,
    )
    dilated = _primitive(
        checked_frame,
        radius=checked_radius,
        shape=checked_shape,
        border=checked_border,
        border_value=checked_border_value,
        dilate=True,
    )
    return _primitive(
        dilated,
        radius=checked_radius,
        shape=checked_shape,
        border=checked_border,
        border_value=checked_border_value,
        dilate=False,
    )


def morphological_gradient(
    frame: Frame,
    *,
    radius: int,
    shape: MorphologyShape = "disk",
    border: Border = "replicate",
    border_value: float | None = None,
) -> Frame:
    """Return dilation minus erosion for an fp32 Frame.

    Both extrema use the same ``radius``, ``disk`` or ``square`` shape, and
    border. Border defaults to neutral ``replicate``; ``constant`` requires a
    finite ``border_value``. The subtraction is per channel, preserves Frame
    metadata, and does not clamp negative values or values above 1.
    """
    checked_frame, checked_radius, checked_shape, checked_border, checked_border_value = _validate_arguments(
        frame,
        operation="morphology.morphological_gradient",
        radius=radius,
        shape=shape,
        border=border,
        border_value=border_value,
    )
    return _gradient_primitive(
        checked_frame,
        radius=checked_radius,
        shape=checked_shape,
        border=checked_border,
        border_value=checked_border_value,
    )


def white_tophat(
    frame: Frame,
    *,
    radius: int,
    shape: MorphologyShape = "disk",
    border: Border = "replicate",
    border_value: float | None = None,
) -> Frame:
    """Return input minus opening to extract small bright detail.

    Opening uses one ``radius``, ``disk`` or ``square`` shape, and border for
    its erosion and dilation. Border defaults to neutral ``replicate``;
    ``constant`` requires finite ``border_value``. The fp32 subtraction is per
    channel, preserves Frame metadata, and does not clamp scene values.
    """
    checked_frame, checked_radius, checked_shape, checked_border, checked_border_value = _validate_arguments(
        frame,
        operation="morphology.white_tophat",
        radius=radius,
        shape=shape,
        border=border,
        border_value=border_value,
    )
    eroded = _primitive(
        checked_frame,
        radius=checked_radius,
        shape=checked_shape,
        border=checked_border,
        border_value=checked_border_value,
        dilate=False,
    )
    return _primitive(
        eroded,
        radius=checked_radius,
        shape=checked_shape,
        border=checked_border,
        border_value=checked_border_value,
        dilate=True,
        difference_source=checked_frame.data,
        difference_mode=_DIFFERENCE_SOURCE_MINUS_RESULT,
    )


def black_tophat(
    frame: Frame,
    *,
    radius: int,
    shape: MorphologyShape = "disk",
    border: Border = "replicate",
    border_value: float | None = None,
) -> Frame:
    """Return closing minus input to extract small dark detail.

    Closing uses one ``radius``, ``disk`` or ``square`` shape, and border for
    its dilation and erosion. Border defaults to neutral ``replicate``;
    ``constant`` requires finite ``border_value``. The fp32 subtraction is per
    channel, preserves Frame metadata, and does not clamp scene values.
    """
    checked_frame, checked_radius, checked_shape, checked_border, checked_border_value = _validate_arguments(
        frame,
        operation="morphology.black_tophat",
        radius=radius,
        shape=shape,
        border=border,
        border_value=border_value,
    )
    dilated = _primitive(
        checked_frame,
        radius=checked_radius,
        shape=checked_shape,
        border=checked_border,
        border_value=checked_border_value,
        dilate=True,
    )
    return _primitive(
        dilated,
        radius=checked_radius,
        shape=checked_shape,
        border=checked_border,
        border_value=checked_border_value,
        dilate=False,
        difference_source=checked_frame.data,
        difference_mode=_DIFFERENCE_RESULT_MINUS_SOURCE,
    )
