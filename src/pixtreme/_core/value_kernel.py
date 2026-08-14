"""CUDA kernel composition for value conversion and array repacking."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import cupy as cp

_CUDA_STORAGE_TYPES = {
    "float32": "float",
    "float16": "unsigned short",
    "uint8": "unsigned char",
    "uint16": "unsigned short",
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
