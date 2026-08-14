"""Test-only EXR primitive harness; production code must not import this module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cupy as cp
import numpy as np

from pixtreme._core.errors import _actionable_error
from pixtreme._io.formats.exr.codec_b44 import _b44_plinear_luts_gpu
from pixtreme._io.formats.exr.codec_dwa import _encode_dwa_huffman_chunks_gpu
from pixtreme._io.formats.exr.codec_piz import (
    _encode_piz_huffman_chunks_gpu,
    _piz_forward_wavelet_kernel,
    _piz_inverse_wavelet_kernel,
)
from pixtreme._io.formats.exr.container import (
    _EXR_PXR24_PLANE_COUNTS,
    _EXR_THREADS_PER_BLOCK,
    _ExrContainer,
    _gpu_error,
    _piz_error,
    _piz_uses_w14,
)
from pixtreme._io.formats.exr.selection import (
    _exr_output_dtype,
    _read_exr_custom_cpu_pixels,
    _read_exr_gpu_pixels,
)
from pixtreme._io.models import ImageHeader


def _read_exr_pixels_with_backend(
    _path: Path,
    container: _ExrContainer,
    header: ImageHeader,
    locations: list[tuple[int, str, str]],
    *,
    unchanged: bool,
    backend: str,
) -> cp.ndarray:
    _exr_output_dtype(header, locations, unchanged=unchanged)
    if backend == "gpu":
        return _read_exr_gpu_pixels(container, header, locations, unchanged=unchanged)
    if backend == "custom_cpu":
        return _read_exr_custom_cpu_pixels(container, header, locations, unchanged=unchanged)
    raise RuntimeError(
        _actionable_error(
            why="the internal EXR read route received an unknown backend",
            what=f"backend={backend!r}, compression={container.compression!r}",
            how="force gpu or custom_cpu only from an internal correctness or benchmark harness",
        )
    )


@dataclass(frozen=True)
class _CudaTransfer:
    direction: str
    nbytes: int
    shape: tuple[int, ...]
    dtype: str
    has_output_buffer: bool = False


def _record_cupy_transfers(monkeypatch: Any) -> list[_CudaTransfer]:
    """Record observable host/device array transfers without changing their behavior."""
    records: list[_CudaTransfer] = []
    original_asarray = cp.asarray
    original_asnumpy = cp.asnumpy
    original_get = cp.ndarray.get

    def asarray_spy(value: object, *args: object, **kwargs: object) -> cp.ndarray:
        if not isinstance(value, cp.ndarray) and not hasattr(value, "__cuda_array_interface__"):
            try:
                host = np.asarray(value)
            except (TypeError, ValueError):
                host = np.asarray((), dtype=object)
            if host.dtype != np.dtype(object):
                records.append(
                    _CudaTransfer(
                        direction="h2d",
                        nbytes=int(host.nbytes),
                        shape=tuple(int(size) for size in host.shape),
                        dtype=host.dtype.name,
                    )
                )
        return cast(cp.ndarray, original_asarray(value, *args, **kwargs))

    def asnumpy_spy(array: object, *args: object, **kwargs: object) -> np.ndarray:
        if isinstance(array, cp.ndarray):
            records.append(
                _CudaTransfer(
                    direction="d2h",
                    nbytes=int(array.nbytes),
                    shape=tuple(int(size) for size in array.shape),
                    dtype=array.dtype.name,
                )
            )
        return cast(np.ndarray, original_asnumpy(array, *args, **kwargs))

    def get_spy(array: cp.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        records.append(
            _CudaTransfer(
                direction="d2h",
                nbytes=int(array.nbytes),
                shape=tuple(int(size) for size in array.shape),
                dtype=array.dtype.name,
                has_output_buffer=kwargs.get("out") is not None,
            )
        )
        return cast(np.ndarray, original_get(array, *args, **kwargs))

    monkeypatch.setattr(cp, "asarray", asarray_spy)
    monkeypatch.setattr(cp, "asnumpy", asnumpy_spy)
    monkeypatch.setattr(cp.ndarray, "get", get_spy)
    return records


def _assert_cupy_transfer_budget(
    records: list[_CudaTransfer],
    *,
    direction: str,
    max_count: int,
    max_total_nbytes: int,
    max_shape_elements: int,
) -> None:
    """Assert explicit count, byte, and shape ceilings for one transfer direction."""
    selected = [record for record in records if record.direction == direction]
    shape_elements = []
    for record in selected:
        elements = 1
        for size in record.shape:
            elements *= size
        shape_elements.append(elements)

    assert len(selected) <= max_count
    assert sum(record.nbytes for record in selected) <= max_total_nbytes
    assert max(shape_elements, default=0) <= max_shape_elements


def _b44_plinear_encode_gpu(bits: cp.ndarray) -> cp.ndarray:
    source = cp.ascontiguousarray(bits, dtype=cp.uint16)
    return cast(cp.ndarray, _b44_plinear_luts_gpu()[0][source])


def _b44_plinear_decode_gpu(bits: cp.ndarray) -> cp.ndarray:
    source = cp.ascontiguousarray(bits, dtype=cp.uint16)
    return cast(cp.ndarray, _b44_plinear_luts_gpu()[1][source])


def _encode_dwa_huffman_gpu(symbols: cp.ndarray) -> cp.ndarray:
    values = cp.ascontiguousarray(symbols, dtype=cp.uint16).reshape(-1)
    if not int(values.size):
        return cp.empty(0, dtype=cp.uint8)
    encoded, offsets, sizes, _ = _encode_dwa_huffman_chunks_gpu(
        values,
        cp.zeros(int(values.size), dtype=cp.int32),
        1,
    )
    return cast(cp.ndarray, encoded[offsets[0] : offsets[0] + sizes[0]])


def _decode_pxr24_rows_gpu(planes: cp.ndarray, pixel_type: int) -> cp.ndarray:
    plane_count = _EXR_PXR24_PLANE_COUNTS.get(pixel_type)
    source = cp.ascontiguousarray(planes, dtype=cp.uint8)
    if plane_count is None or source.ndim != 3 or source.shape[1] != plane_count:
        raise _gpu_error(
            why="the PXR24 row decoder received an unsupported pixel type or plane shape",
            what=f"pixel_type={pixel_type}, shape={source.shape!r}",
            how="provide row-major UINT=4, HALF=2, or FLOAT24=3 byte planes",
        )
    difference = cp.zeros((source.shape[0], source.shape[2]), dtype=cp.uint32)
    for plane in range(plane_count):
        difference |= source[:, plane].astype(cp.uint32) << cp.uint32(8 * (plane_count - plane - 1))
    mask = cp.uint32((1 << (plane_count * 8)) - 1)
    values = cp.cumsum(difference, axis=1, dtype=cp.uint32) & mask
    return cast(cp.ndarray, values << cp.uint32(8) if pixel_type == 2 else values)


def _encode_piz_huffman_gpu(symbols: cp.ndarray) -> cp.ndarray:
    values = cp.ascontiguousarray(symbols, dtype=cp.uint16).reshape(-1)
    encoded, offsets, sizes, _ = _encode_piz_huffman_chunks_gpu(
        values,
        cp.zeros(int(values.size), dtype=cp.int32),
        1,
    )
    return cast(cp.ndarray, encoded[offsets[0] : offsets[0] + sizes[0]])


def _piz_forward_wavelet_gpu(
    words: cp.ndarray,
    *,
    nx: int,
    ny: int,
    word_stride: int,
    word_slice: int,
    max_value: int,
    word_offset: int = 0,
) -> None:
    values = cp.asarray(words)
    if values.dtype != cp.uint16 or values.ndim != 1 or not values.flags.c_contiguous:
        raise _gpu_error(
            why="the GPU PIZ forward-wavelet input is not a contiguous uint16 word plane",
            what=f"dtype={values.dtype}, shape={values.shape!r}, contiguous={values.flags.c_contiguous}",
            how="stage channel-major low/high word fields contiguously before transformation",
        )
    if nx < 1 or ny < 1 or word_stride < 1 or not 0 <= word_slice < word_stride or word_offset < 0:
        raise _gpu_error(
            why="the GPU PIZ forward-wavelet field geometry is invalid",
            what=(f"nx={nx}, ny={ny}, word_stride={word_stride}, word_slice={word_slice}, word_offset={word_offset}"),
            how="encode a positive sampled plane and an in-range independent word slice",
        )
    required = word_offset + word_slice + (ny - 1) * word_stride * nx + (nx - 1) * word_stride + 1
    if required > int(values.size):
        raise _gpu_error(
            why="the GPU PIZ forward-wavelet field extends beyond its owning word plane",
            what=f"required={required}, words={values.size}, nx={nx}, ny={ny}",
            how="match each field geometry and stride to its channel-plane write descriptor",
        )
    if min(nx, ny) < 2:
        return
    w14 = np.int32(_piz_uses_w14(max_value))
    base = np.int64(word_offset + word_slice)
    p = 1
    p2 = 2
    while p2 <= min(nx, ny):
        columns = nx // p2
        rows = ny // p2
        full_count = columns * rows
        vertical_count = rows if nx & p else 0
        horizontal_count = columns if ny & p else 0
        task_count = full_count + vertical_count + horizontal_count
        if task_count:
            _piz_forward_wavelet_kernel()(
                ((task_count + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK,),
                (_EXR_THREADS_PER_BLOCK,),
                (
                    values,
                    base,
                    np.int32(nx),
                    np.int32(ny),
                    np.int32(word_stride),
                    np.int32(p),
                    w14,
                    np.int64(full_count),
                    np.int64(vertical_count),
                    np.int64(horizontal_count),
                ),
            )
        p = p2
        p2 *= 2


def _piz_inverse_wavelet_gpu(
    words: cp.ndarray,
    *,
    nx: int,
    ny: int,
    word_stride: int,
    word_slice: int,
    max_value: int,
) -> None:
    if words.dtype != cp.uint16 or words.ndim != 1 or not words.flags.c_contiguous:
        raise _piz_error(
            why="the GPU PIZ inverse-wavelet input is not a contiguous uint16 word plane",
            what=f"dtype={words.dtype}, shape={words.shape}, contiguous={words.flags.c_contiguous}",
            how="materialize each Huffman chunk as one contiguous uint16 word vector",
        )
    if nx < 1 or ny < 1 or word_stride < 1 or not 0 <= word_slice < word_stride:
        raise _piz_error(
            why="the GPU PIZ inverse-wavelet field geometry is invalid",
            what=f"nx={nx}, ny={ny}, word_stride={word_stride}, word_slice={word_slice}",
            how="decode a positive sampled plane and an in-range independent word slice",
        )
    required = word_slice + (ny - 1) * word_stride * nx + (nx - 1) * word_stride + 1
    if required > int(words.size):
        raise _piz_error(
            why="the GPU PIZ inverse-wavelet field extends beyond its owning word plane",
            what=f"required={required}, words={int(words.size)}, nx={nx}, ny={ny}",
            how="match each field geometry and stride to its channel-plane descriptor",
        )
    if min(nx, ny) < 2:
        return
    p2 = 1 << (min(nx, ny).bit_length() - 1)
    p = p2 // 2
    while p >= 1:
        step = p * 2
        columns = nx // step
        rows = ny // step
        full_count = columns * rows
        vertical_count = rows if nx & p else 0
        horizontal_count = columns if ny & p else 0
        task_count = full_count + vertical_count + horizontal_count
        if task_count:
            grid = (task_count + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK
            _piz_inverse_wavelet_kernel()(
                (grid,),
                (_EXR_THREADS_PER_BLOCK,),
                (
                    words,
                    np.int64(word_slice),
                    np.int32(nx),
                    np.int32(ny),
                    np.int32(word_stride),
                    np.int32(p),
                    np.int32(_piz_uses_w14(max_value)),
                    np.int64(full_count),
                    np.int64(vertical_count),
                    np.int64(horizontal_count),
                ),
            )
        p //= 2
