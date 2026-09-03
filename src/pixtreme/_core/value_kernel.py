"""CUDA kernel composition for value conversion and array repacking."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import cupy as cp
import numpy as np

_CUDA_STORAGE_TYPES = {
    "float32": "float",
    "float16": "unsigned short",
    "uint8": "unsigned char",
    "uint16": "unsigned short",
    "uint32": "unsigned int",
}


def _cuda_storage_type(dtype: str) -> str:
    return _CUDA_STORAGE_TYPES[dtype]


def _cuda_load_expression(dtype: str, value: str) -> str:
    if dtype == "float32":
        return value
    if dtype == "float16":
        return f"__half2float(__ushort_as_half({value}))"
    return f"(float)({value})"


def _cuda_store_expression(dtype: str, value: str) -> str:
    if dtype == "float32":
        return value
    if dtype == "float16":
        return f"__half_as_ushort(__float2half_rn({value}))"
    return f"({_cuda_storage_type(dtype)})({value})"


def _quantize_expression(value: str) -> str:
    return f"floorf(fminf(fmaxf({value}, 0.0f), 1.0f) * value_maximum + 0.5f)"


def _dequantize_expression(value: str) -> str:
    return f"({value} * value_scale)"


def _affine_parameter_declarations(channel_count: int, *, scalar_affine: bool) -> str:
    if not scalar_affine:
        return (
            "const float* __restrict__ scale,\n    const float* __restrict__ mean,\n    const float* __restrict__ std"
        )
    return ",\n    ".join(
        f"const float {name}_{channel}" for name in ("scale", "mean", "std") for channel in range(channel_count)
    )


def _affine_operand(name: str, channel: int, *, scalar_affine: bool) -> str:
    return f"{name}_{channel}" if scalar_affine else f"{name}[{channel}]"


@lru_cache(maxsize=16)
def _linear_value_kernel(
    kernel_name: str,
    source_dtype: str,
    destination_dtype: str,
    transform: Callable[[str], str],
    parameter_name: str,
) -> cp.RawKernel:
    loaded = _cuda_load_expression(source_dtype, "source[element]")
    stored = _cuda_store_expression(destination_dtype, transform(loaded))
    source = f"""
#include <cuda_fp16.h>

extern "C" __global__ void {kernel_name}(
    const {_cuda_storage_type(source_dtype)}* __restrict__ source,
    {_cuda_storage_type(destination_dtype)}* __restrict__ destination,
    const long long element_count,
    const float {parameter_name}
) {{
    const long long element = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= element_count) {{
        return;
    }}
    destination[element] = {stored};
}}
"""
    return cp.RawKernel(source, kernel_name)


@lru_cache(maxsize=128)
def _from_array_kernel(
    source_dtype: str,
    destination_dtype: str,
    channel_count: int,
    index_mode: str,
    scalar_affine: bool,
    dequantize: bool,
) -> cp.RawKernel:
    assignments: list[str] = []
    for channel in range(channel_count):
        if index_mode == "hwc":
            source_index = f"pixel * {channel_count} + {channel}"
        elif index_mode == "chw":
            source_index = f"pixel + {channel} * pixel_count"
        else:
            source_index = f"source_base + {channel} * channel_stride"
        loaded = _cuda_load_expression(source_dtype, f"source[{source_index}]")
        if dequantize:
            value = _dequantize_expression(loaded)
        else:
            scale = _affine_operand("scale", channel, scalar_affine=scalar_affine)
            mean = _affine_operand("mean", channel, scalar_affine=scalar_affine)
            std = _affine_operand("std", channel, scalar_affine=scalar_affine)
            value = f"(({loaded} * {std} + {mean}) / {scale})"
        stored = _cuda_store_expression(destination_dtype, value)
        assignments.append(f"    destination[pixel * {channel_count} + {channel}] = {stored};")

    strided_prelude = ""
    if index_mode == "strided":
        strided_prelude = """
    const long long row = pixel / width;
    const long long column = pixel - row * width;
    const long long source_base = row * row_stride + column * column_stride;
"""
    affine_parameters = _affine_parameter_declarations(channel_count, scalar_affine=scalar_affine)
    source = f"""
#include <cuda_fp16.h>

extern "C" __global__ void pixtreme_from_array_affine(
    const {_cuda_storage_type(source_dtype)}* __restrict__ source,
    {_cuda_storage_type(destination_dtype)}* __restrict__ destination,
    const long long pixel_count,
    const long long width,
    const long long row_stride,
    const long long column_stride,
    const long long channel_stride,
    const float value_scale,
    {affine_parameters}
) {{
    const long long pixel = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (pixel >= pixel_count) {{
        return;
    }}
{strided_prelude}
{chr(10).join(assignments)}
}}
"""
    return cp.RawKernel(source, "pixtreme_from_array_affine")


@lru_cache(maxsize=256)
def _to_array_kernel(
    source_dtype: str,
    destination_dtype: str,
    source_channels: int,
    channel_indices: tuple[int, ...],
    channels_first: bool,
    scalar_affine: bool,
    quantize: bool,
) -> cp.RawKernel:
    output_channels = len(channel_indices)
    assignments: list[str] = []
    for output_channel, source_channel in enumerate(channel_indices):
        source_index = f"pixel * {source_channels} + {source_channel}"
        destination_index = (
            f"{output_channel} * pixel_count + pixel"
            if channels_first
            else f"pixel * {output_channels} + {output_channel}"
        )
        loaded = _cuda_load_expression(source_dtype, f"source[{source_index}]")
        if quantize:
            value = _quantize_expression(loaded)
        else:
            scale = _affine_operand("scale", output_channel, scalar_affine=scalar_affine)
            mean = _affine_operand("mean", output_channel, scalar_affine=scalar_affine)
            std = _affine_operand("std", output_channel, scalar_affine=scalar_affine)
            value = f"(({loaded} * {scale} - {mean}) / {std})"
        stored = _cuda_store_expression(destination_dtype, value)
        assignments.append(f"    destination[{destination_index}] = {stored};")

    affine_parameters = _affine_parameter_declarations(output_channels, scalar_affine=scalar_affine)
    source = f"""
#include <cuda_fp16.h>

extern "C" __global__ void pixtreme_to_array_affine(
    const {_cuda_storage_type(source_dtype)}* __restrict__ source,
    {_cuda_storage_type(destination_dtype)}* __restrict__ destination,
    const long long pixel_count,
    const float value_maximum,
    {affine_parameters}
) {{
    const long long pixel = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (pixel >= pixel_count) {{
        return;
    }}
{chr(10).join(assignments)}
}}
"""
    return cp.RawKernel(source, "pixtreme_to_array_affine")


_SCALAR_ROUTE_SOURCE_LIMIT = 32
_SCALAR_ROUTE_OUTPUT_LIMIT = 32


@lru_cache(maxsize=128)
def _route_float32_channels_kernel(source_count: int, output_count: int) -> cp.RawKernel:
    source_parameters = [
        f"const unsigned int* __restrict__ source_{source_index}" for source_index in range(source_count)
    ]
    source_channel_parameters = [f"const int source_channels_{source_index}" for source_index in range(source_count)]
    route_parameters = [
        parameter
        for output_index in range(output_count)
        for parameter in (
            f"const int route_source_{output_index}",
            f"const int route_channel_{output_index}",
            f"const unsigned int fill_bits_{output_index}",
        )
    ]
    parameters = ",\n    ".join(
        (
            *source_parameters,
            "unsigned int* __restrict__ destination",
            "const long long pixel_count",
            *source_channel_parameters,
            *route_parameters,
        )
    )

    assignments: list[str] = []
    for output_index in range(output_count):
        source_cases = "\n".join(
            (
                f"            case {source_index}: value_bits = "
                f"source_{source_index}[pixel * (long long)source_channels_{source_index} "
                f"+ route_channel_{output_index}]; break;"
            )
            for source_index in range(source_count)
        )
        assignments.append(
            f"""
    {{
        unsigned int value_bits = fill_bits_{output_index};
        if (route_source_{output_index} >= 0) {{
            switch (route_source_{output_index}) {{
{source_cases}
            default: break;
            }}
        }}
        destination[pixel * {output_count} + {output_index}] = value_bits;
    }}"""
        )

    source = f"""
extern "C" __global__ void pixtreme_route_float32_channels(
    {parameters}
) {{
    const long long pixel = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (pixel >= pixel_count) {{
        return;
    }}
{"".join(assignments)}
}}
"""
    return cp.RawKernel(source, "pixtreme_route_float32_channels")


@lru_cache(maxsize=1)
def _route_float32_channels_descriptor_kernel() -> cp.RawKernel:
    source = r"""
extern "C" __global__ void pixtreme_route_float32_channels_descriptor(
    const unsigned long long* __restrict__ descriptors,
    unsigned int* __restrict__ destination,
    const long long element_count,
    const int source_count,
    const int output_count
) {
    const long long element = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= element_count) {
        return;
    }

    const long long pixel = element / output_count;
    const int output_channel = (int)(element - pixel * output_count);
    const long long route_offset = 2LL * source_count + 2LL * output_channel;
    const unsigned long long route = descriptors[route_offset];
    const unsigned int source_code = (unsigned int)(route >> 32);
    const unsigned int source_channel = (unsigned int)route;
    unsigned int value_bits = (unsigned int)descriptors[route_offset + 1];

    if (source_code != 0) {
        const unsigned int source_index = source_code - 1;
        const unsigned long long source_pointer = descriptors[2LL * source_index];
        const unsigned long long source_channels = descriptors[2LL * source_index + 1];
        const unsigned int* source = reinterpret_cast<const unsigned int*>(source_pointer);
        value_bits = source[pixel * source_channels + source_channel];
    }
    destination[element] = value_bits;
}
"""
    return cp.RawKernel(source, "pixtreme_route_float32_channels_descriptor")


def _float32_fill_bits(value: np.float32) -> np.uint32:
    return np.uint32(np.asarray(value, dtype=np.float32).view(np.uint32).item())


def _route_float32_channels(
    sources: tuple[cp.ndarray, ...],
    routes: tuple[tuple[int, int] | np.float32, ...],
) -> cp.ndarray:
    """Assemble float32 HWC source channels and fill values in one GPU pass."""
    height, width = sources[0].shape[:2]
    pixel_count = height * width
    output = cp.empty((height, width, len(routes)), dtype=cp.float32)

    if len(sources) <= _SCALAR_ROUTE_SOURCE_LIMIT and len(routes) <= _SCALAR_ROUTE_OUTPUT_LIMIT:
        arguments: list[object] = [
            *sources,
            output,
            np.int64(pixel_count),
            *(np.int32(source.shape[2]) for source in sources),
        ]
        for route in routes:
            if isinstance(route, tuple):
                source_index, channel_index = route
                arguments.extend((np.int32(source_index), np.int32(channel_index), np.uint32(0)))
            else:
                arguments.extend((np.int32(-1), np.int32(0), _float32_fill_bits(route)))
        _route_float32_channels_kernel(len(sources), len(routes))(
            ((pixel_count + 255) // 256,),
            (256,),
            tuple(arguments),
        )
        return output

    descriptor_count = 2 * len(sources) + 2 * len(routes)
    host_descriptors = np.empty(descriptor_count, dtype=np.uint64)
    for source_index, source in enumerate(sources):
        host_descriptors[2 * source_index] = np.uint64(source.data.ptr)
        host_descriptors[2 * source_index + 1] = np.uint64(source.shape[2])
    route_offset = 2 * len(sources)
    for output_index, route in enumerate(routes):
        if isinstance(route, tuple):
            source_index, channel_index = route
            route_value = ((source_index + 1) << 32) | channel_index
            fill_bits = np.uint32(0)
        else:
            route_value = 0
            fill_bits = _float32_fill_bits(route)
        host_descriptors[route_offset + 2 * output_index] = np.uint64(route_value)
        host_descriptors[route_offset + 2 * output_index + 1] = np.uint64(fill_bits)

    descriptors = cp.asarray(host_descriptors)
    element_count = pixel_count * len(routes)
    _route_float32_channels_descriptor_kernel()(
        ((element_count + 255) // 256,),
        (256,),
        (descriptors, output, np.int64(element_count), np.int32(len(sources)), np.int32(len(routes))),
    )
    return output
