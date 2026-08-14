"""In-memory CUDA array import, export, and affine repacking."""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol, cast

import cupy as cp
import numpy as np

from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import (
    _ACCEPTED_DTYPES,
    _COLORSPACE_TOKENS,
    _DTYPE_TOKENS,
    _GAMMA_TOKENS,
    _LAYOUT_TOKENS,
    ChannelInput,
    Frame,
    _normalize_channels,
    _validate_token,
)
from pixtreme._core.value_domain import (
    _bit_depth_maximum,
    _bit_depth_scale,
    _container_dtype,
    _float32_conversion_guidance,
    _validate_bit_depth,
)
from pixtreme._core.value_kernel import _from_array_kernel, _to_array_kernel

_DLPACK_DEVICE_CUDA = 2
_FROM_ARRAY_THREADS_PER_BLOCK = 512
_TO_ARRAY_THREADS_PER_BLOCK = 1024
_SCALAR_AFFINE_CHANNEL_LIMIT = 16


class _DLPackProducer(Protocol):
    def __dlpack__(self, *, stream: int | None = None) -> object: ...

    def __dlpack_device__(self) -> tuple[int, int]: ...


def to_array(
    frame: Frame,
    *,
    channels: ChannelInput | None = None,
    layout: str | None = None,
    dtype: str | None = None,
    bit_depth: int | None = None,
    scale: object | None = None,
    mean: object | None = None,
    std: object | None = None,
    out: object | None = None,
    copy: bool | None = None,
) -> cp.ndarray:
    """Export this Frame to a CuPy device array, optionally in one fused affine pass.

    ``channels`` selects and orders labels. ``layout`` is HWC, NHWC, CHW,
    or NCHW. The affine formula is ``y = (x * scale - mean) / std`` and is
    the inverse of :func:`from_array` when the same constants are used.
    Constants may be scalars or one value per exported channel; arithmetic
    is fp32 before the faithful ``dtype`` cast.

    ``bit_depth`` selects the 8, 10, 12, 14, or 16-bit unsigned full-scale
    grid. It clips fp32 values to ``[0, 1]``, scales by
    ``2^bit_depth - 1``, and rounds half away from zero. The output dtype is
    uint8 for 8-bit codes and uint16 otherwise. It cannot be combined with
    affine constants. Use :func:`quantize` when the result should
    remain a metadata-bearing Frame.

    ``copy=None`` uses a zero-copy view when possible and otherwise makes
    one fused copy. ``copy=False`` strictly requires zero-copy and raises
    when repacking is unavoidable. ``copy=True`` always returns private
    storage. With ``out``, copy must be omitted and the destination must be
    an exactly shaped, exactly typed, C-contiguous ``cupy.ndarray``; the
    fused pass writes there directly and returns that same object. A
    non-CuPy DLPack producer is intentionally rejected as a destination;
    explicitly create a writable view with ``cp.from_dlpack(tensor)`` and
    pass that CuPy array instead.

    The returned ``cupy.ndarray`` is itself a DLPack producer. Frame is also
    a DLPack producer through its protocol methods; no to_tensor or
    to_dlpack helper is needed.
    """
    _validate_copy(copy)
    layout_token = _validate_layout(layout)
    requested_channels, indices = _select_channel_indices(frame.channels, channels)
    validated_bit_depth = None if bit_depth is None else _validate_bit_depth(bit_depth)
    affine_requested = any(value is not None for value in (scale, mean, std))
    if validated_bit_depth is not None:
        if affine_requested:
            raise ValueError(
                _actionable_error(
                    why="to_array bit_depth quantization cannot share a pass with affine scale, mean, or std",
                    what=(f"received bit_depth={validated_bit_depth}, scale={scale!r}, mean={mean!r}, std={std!r}"),
                    how=f"call to_array(bit_depth={validated_bit_depth}) without scale, mean, or std",
                )
            )
        source_dtype = np.dtype(frame.data.dtype)
        if source_dtype != np.dtype(np.float32):
            raise ValueError(
                _actionable_error(
                    why="to_array bit_depth quantization requires normalized float32 Frame data",
                    what=f"received Frame data dtype={frame.data.dtype!s} with bit_depth={validated_bit_depth}",
                    how=f"{_float32_conversion_guidance(source_dtype)}, then call to_array(bit_depth={validated_bit_depth})",
                )
            )
        container_dtype = _container_dtype(validated_bit_depth)
        output_dtype = _resolve_dtype(dtype, default=container_dtype)
        if output_dtype != container_dtype:
            raise ValueError(
                _actionable_error(
                    why=f"to_array bit_depth={validated_bit_depth} selects {container_dtype.name} output storage",
                    what=f"received dtype={dtype!r}, resolved to {output_dtype.name!r}",
                    how=f"omit dtype or pass dtype={container_dtype.name!r}",
                )
            )
    else:
        output_dtype = _resolve_dtype(dtype, default=np.dtype(frame.data.dtype))
    output_shape = _layout_shape(
        height=frame.height,
        width=frame.width,
        channel_count=len(requested_channels),
        layout=layout_token,
    )

    if out is not None and copy is not None:
        raise ValueError(
            _actionable_error(
                why="out already supplies destination ownership, so copy has no defined meaning",
                what=f"copy={copy!r} was provided together with out",
                how="omit copy when passing out",
            )
        )

    destination = _validate_out(out, expected_shape=output_shape, expected_dtype=output_dtype)
    channel_repacking = indices != tuple(range(len(frame.channels)))
    layout_repacking = layout_token in {"CHW", "NCHW"}
    dtype_repacking = output_dtype != np.dtype(frame.data.dtype)
    requires_copy = (
        channel_repacking or layout_repacking or dtype_repacking or affine_requested or validated_bit_depth is not None
    )

    if destination is None and not requires_copy:
        view = frame.data if layout_token == "HWC" else frame.data.reshape((1, *frame.data.shape))
        return view.copy() if copy is True else view

    if destination is None and copy is False:
        raise ValueError(
            _actionable_error(
                why="the requested channel, layout, dtype, bit_depth, or affine conversion requires a write",
                what="copy=False requires a zero-copy to_array result",
                how="use copy=None or copy=True, or omit the repacking arguments",
            )
        )

    if destination is None:
        destination = cp.empty(output_shape, dtype=output_dtype)
    channel_count = len(indices)
    affine_scale = _affine_values(scale, name="scale", channel_count=channel_count, default=1.0)
    affine_mean = _affine_values(mean, name="mean", channel_count=channel_count, default=0.0)
    affine_std = _affine_values(std, name="std", channel_count=channel_count, default=1.0)
    scalar_affine = channel_count <= _SCALAR_AFFINE_CHANNEL_LIMIT
    affine_arguments = _affine_kernel_arguments(
        affine_scale,
        affine_mean,
        affine_std,
        scalar_affine=scalar_affine,
    )
    pixel_count = frame.width * frame.height
    if pixel_count > 0:
        block_count = (pixel_count + _TO_ARRAY_THREADS_PER_BLOCK - 1) // _TO_ARRAY_THREADS_PER_BLOCK
        _to_array_kernel(
            np.dtype(frame.data.dtype).name,
            output_dtype.name,
            int(frame.data.shape[2]),
            indices,
            layout_token in {"CHW", "NCHW"},
            scalar_affine,
            validated_bit_depth is not None,
        )(
            (block_count,),
            (_TO_ARRAY_THREADS_PER_BLOCK,),
            (
                frame.data,
                destination,
                np.int64(pixel_count),
                _bit_depth_maximum(validated_bit_depth) if validated_bit_depth is not None else np.float32(1.0),
                *affine_arguments,
            ),
        )
    return destination


def _from_cuda_dlpack(data: object) -> cp.ndarray:
    device_query = getattr(data, "__dlpack_device__", None)
    exporter = getattr(data, "__dlpack__", None)
    if not callable(device_query) or not callable(exporter):
        raise ValueError(
            _actionable_error(
                why="from_array accepts device-resident arrays only",
                what=f"received {type(data).__module__}.{type(data).__qualname__}, not a CUDA DLPack producer",
                how="move host data explicitly with cp.asarray(data), then call px.io.from_array",
            )
        )

    producer = cast(_DLPackProducer, data)
    try:
        device_type, _ = producer.__dlpack_device__()
    except Exception as error:
        raise ValueError(
            _actionable_error(
                why="the DLPack producer device could not be queried",
                what=type(data).__qualname__,
                how="provide a CUDA producer with working __dlpack__ and __dlpack_device__ methods",
            )
        ) from error
    if device_type != _DLPACK_DEVICE_CUDA:
        raise ValueError(
            _actionable_error(
                why="from_array accepts CUDA DLPack producers only",
                what=f"producer reported DLPack device type {device_type}",
                how="move data to CUDA explicitly with cp.asarray(data) or tensor.to('cuda') before px.io.from_array",
            )
        )

    try:
        return cast(cp.ndarray, cp.from_dlpack(producer))
    except Exception as error:
        raise ValueError(
            _actionable_error(
                why="CuPy could not import the CUDA producer through DLPack",
                what=type(data).__qualname__,
                how="verify the producer stream and lifetime, then pass a valid CUDA DLPack producer",
            )
        ) from error


def _validate_copy(copy: bool | None) -> None:
    if copy is not None and type(copy) is not bool:
        raise ValueError(
            _actionable_error(
                why="copy uses the DLPack-style None/False/True contract",
                what=f"received copy={copy!r}",
                how="pass None, False, or True",
            )
        )


def _validate_layout(layout: str | None) -> str:
    token = "HWC" if layout is None else layout
    if token not in _LAYOUT_TOKENS:
        raise ValueError(
            _actionable_error(
                why="layout accepts only canonical, case-sensitive axis-order tokens",
                what=f"received layout={token!r}",
                how=f"pass layout=<token> with one of {_LAYOUT_TOKENS!r}",
            )
        )
    return token


def _resolve_dtype(dtype: str | None, *, default: np.dtype[np.generic]) -> np.dtype[np.generic]:
    if dtype is None:
        return default
    if not isinstance(dtype, str):
        raise ValueError(
            _actionable_error(
                why="dtype must be a string token so the requested storage type is unambiguous",
                what=f"received dtype={dtype!r} ({type(dtype).__module__}.{type(dtype).__qualname__})",
                how=f"pass dtype=<token> with one of {tuple(_DTYPE_TOKENS)!r}",
            )
        )
    resolved = _DTYPE_TOKENS.get(dtype)
    if resolved is None:
        raise ValueError(
            _actionable_error(
                why="dtype accepts only supported canonical storage tokens",
                what=f"received dtype={dtype!r}",
                how=f"pass dtype=<token> with one of {tuple(_DTYPE_TOKENS)!r}",
            )
        )
    return cast(np.dtype[np.generic], resolved)


def _as_hwc(array: cp.ndarray, *, layout: str) -> cp.ndarray:
    if layout == "HWC":
        if array.ndim != 3:
            raise ValueError(
                _actionable_error(
                    why="layout HWC declares rank 3 data ordered as height, width, channels",
                    what=f"received layout='HWC' with shape={array.shape!r}",
                    how="reshape data to (height, width, channels), or pass the layout token matching its current axes",
                )
            )
        return array
    if layout == "NHWC":
        if array.ndim != 4:
            raise ValueError(
                _actionable_error(
                    why="layout NHWC declares rank 4 data ordered as batch, height, width, channels",
                    what=f"received layout='NHWC' with shape={array.shape!r}",
                    how="reshape data to (1, height, width, channels), or pass layout='HWC' for rank 3 HWC data",
                )
            )
        if array.shape[0] != 1:
            raise ValueError(
                _actionable_error(
                    why="layout NHWC supports a single image with N == 1",
                    what=f"received layout='NHWC' with N={array.shape[0]} and shape={array.shape!r}",
                    how="select one batch item so data.shape[0] == 1 before calling from_array",
                )
            )
        return array.reshape(array.shape[1:])
    if layout == "CHW":
        if array.ndim != 3:
            raise ValueError(
                _actionable_error(
                    why="layout CHW declares rank 3 data ordered as channels, height, width",
                    what=f"received layout='CHW' with shape={array.shape!r}",
                    how="reshape data to (channels, height, width), or pass the layout token matching its current axes",
                )
            )
        return array.transpose(1, 2, 0)
    if array.ndim != 4:
        raise ValueError(
            _actionable_error(
                why="layout NCHW declares rank 4 data ordered as batch, channels, height, width",
                what=f"received layout='NCHW' with shape={array.shape!r}",
                how="reshape data to (1, channels, height, width), or pass layout='CHW' for rank 3 CHW data",
            )
        )
    if array.shape[0] != 1:
        raise ValueError(
            _actionable_error(
                why="layout NCHW supports a single image with N == 1",
                what=f"received layout='NCHW' with N={array.shape[0]} and shape={array.shape!r}",
                how="select one batch item so data.shape[0] == 1 before calling from_array",
            )
        )
    return array[0].transpose(1, 2, 0)


def _affine_values(value: object | None, *, name: str, channel_count: int, default: float) -> tuple[np.float32, ...]:
    if value is None:
        return (np.float32(default),) * channel_count
    if isinstance(value, (str, bytes)):
        raise ValueError(
            _actionable_error(
                why=f"{name} must be numeric; strings and bytes are not affine constants",
                what=f"received {name}={value!r}",
                how=f"pass {name}=<number> or a sequence of {channel_count} numeric values",
            )
        )
    if np.isscalar(value):
        try:
            scalar = np.float32(cast(float | int, value))
        except (TypeError, ValueError) as error:
            raise ValueError(
                _actionable_error(
                    why=f"{name} scalar could not be converted to float32",
                    what=f"received {name}={value!r}",
                    how=f"pass {name}=<number> or a sequence of {channel_count} numeric values",
                )
            ) from error
        return (scalar,) * channel_count
    if isinstance(value, tuple) and len(value) == channel_count:
        try:
            return tuple(np.float32(component) for component in value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                _actionable_error(
                    why=f"one or more {name} sequence entries could not be converted to float32",
                    what=f"received {name}={value!r}",
                    how=f"pass exactly {channel_count} numeric values for {name}",
                )
            ) from error
    try:
        candidate = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(
            _actionable_error(
                why=f"{name} could not be converted to a float32 scalar or array",
                what=f"received {name}={value!r}",
                how=f"pass {name}=<number> or a sequence of {channel_count} numeric values",
            )
        ) from error
    if candidate.ndim == 0:
        return (np.float32(candidate.item()),) * channel_count
    if candidate.ndim == 1 and candidate.size == channel_count:
        host = np.ascontiguousarray(candidate)
    else:
        raise ValueError(
            _actionable_error(
                why=f"{name} must provide one scalar or exactly one value per exported channel",
                what=f"received {name}={value!r} with shape={candidate.shape!r} for {channel_count} channels",
                how=f"pass {name}=<number> or a one-dimensional sequence of {channel_count} values",
            )
        )
    return tuple(np.float32(component) for component in host)


def _affine_kernel_arguments(
    scale: tuple[np.float32, ...],
    mean: tuple[np.float32, ...],
    std: tuple[np.float32, ...],
    *,
    scalar_affine: bool,
) -> tuple[object, ...]:
    if scalar_affine:
        return (*scale, *mean, *std)
    return (
        cp.asarray(np.asarray(scale, dtype=np.float32)),
        cp.asarray(np.asarray(mean, dtype=np.float32)),
        cp.asarray(np.asarray(std, dtype=np.float32)),
    )


def _select_channel_indices(
    source_channels: tuple[str, ...], requested: ChannelInput | None
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if requested is None:
        return source_channels, tuple(range(len(source_channels)))
    normalized = _normalize_channels(requested)
    return normalized, _select_normalized_channel_indices(source_channels, normalized)


@lru_cache(maxsize=256)
def _select_normalized_channel_indices(
    source_channels: tuple[str, ...], normalized: tuple[str, ...]
) -> tuple[int, ...]:
    remaining = list(enumerate(source_channels))
    indices: list[int] = []
    for label in normalized:
        match = next((index for index, (_, candidate) in enumerate(remaining) if candidate == label), None)
        if match is None:
            raise ValueError(
                _actionable_error(
                    why=f"requested channel {label!r} is not available after earlier channel selections",
                    what=f"requested channels={normalized!r} from Frame channels={source_channels!r}",
                    how=f"request each label no more times than it appears in channels={source_channels!r}",
                )
            )
        source_index, _ = remaining.pop(match)
        indices.append(source_index)
    return tuple(indices)


def _layout_shape(*, height: int, width: int, channel_count: int, layout: str) -> tuple[int, ...]:
    if layout == "HWC":
        return (height, width, channel_count)
    if layout == "NHWC":
        return (1, height, width, channel_count)
    if layout == "CHW":
        return (channel_count, height, width)
    return (1, channel_count, height, width)


def _validate_out(
    out: object | None, *, expected_shape: tuple[int, ...], expected_dtype: np.dtype[np.generic]
) -> cp.ndarray | None:
    if out is None:
        return None
    if not isinstance(out, cp.ndarray):
        raise ValueError(
            _actionable_error(
                why="out must prove zero-copy writable CuPy storage",
                what=f"received {type(out).__module__}.{type(out).__qualname__}",
                how="pass a cupy.ndarray; for another DLPack producer, first use cp.from_dlpack(tensor)",
            )
        )
    if out.shape != expected_shape:
        raise ValueError(
            _actionable_error(
                why="out shape must exactly match the requested to_array result",
                what=f"expected {expected_shape!r}, got {out.shape!r}",
                how=f"allocate cp.empty({expected_shape!r}, dtype={expected_dtype.name!r})",
            )
        )
    if np.dtype(out.dtype) != expected_dtype:
        raise ValueError(
            _actionable_error(
                why="out dtype must exactly match the requested to_array dtype",
                what=f"expected {expected_dtype.name}, got {out.dtype!s}",
                how=f"allocate out with dtype={expected_dtype.name!r} or change the dtype argument",
            )
        )
    if not out.flags.c_contiguous:
        raise ValueError(
            _actionable_error(
                why="the fused export kernel requires a C-contiguous destination",
                what=f"out with shape {out.shape!r} is not C-contiguous",
                how="pass cp.ascontiguousarray(out) or allocate a C-contiguous cupy.ndarray",
            )
        )
    return out


def _from_array_index_mode(logical: cp.ndarray) -> tuple[str, tuple[int, int, int]]:
    itemsize = int(logical.dtype.itemsize)
    strides = cast(tuple[int, int, int], tuple(int(stride // itemsize) for stride in logical.strides))
    height, width, channel_count = (int(size) for size in logical.shape)
    if strides == (width * channel_count, channel_count, 1):
        return "hwc", strides
    if strides == (width, 1, height * width):
        return "chw", strides
    return "strided", strides


def from_array(
    data: object,
    *,
    colorspace: str,
    gamma: str,
    channels: ChannelInput,
    matrix: str | None = None,
    layout: str | None = None,
    dtype: str | None = None,
    bit_depth: int | None = None,
    scale: object | None = None,
    mean: object | None = None,
    std: object | None = None,
    copy: bool | None = None,
) -> Frame:
    """Construct a Frame from a CuPy array or CUDA DLPack producer.

    ``layout`` declares HWC, NHWC, CHW, or NCHW input. ``dtype`` selects one of
        the five Frame storage dtypes. The inverse affine formula is
    ``x = (y * std + mean) / scale``; it round-trips :func:`pixtreme.io.to_array`
    with the same scalar or per-channel constants and computes in fp32.

    ``bit_depth`` selects the 8, 10, 12, 14, or 16-bit unsigned full-scale grid
    and normalizes a matching uint container by ``2^bit_depth - 1`` into fp32.
    It cannot be combined with affine constants; explicit ``dtype`` must be
    ``"float32"``.

    ``copy=None`` retains a zero-copy HWC/NHWC view when possible and otherwise
    performs one fused copy. ``copy=False`` strictly guarantees zero-copy and
    raises if contiguity, layout, dtype, or affine processing requires a write.
    ``copy=True`` always gives the Frame private storage. Host arrays and CPU
    DLPack producers are rejected rather than transferred implicitly.
    ``matrix`` stamps YCbCr basis provenance only; it never changes array
    values, ownership, layout, dtype, or affine processing.
    """
    _validate_copy(copy)
    array = data if isinstance(data, cp.ndarray) else _from_cuda_dlpack(data)
    layout_token = _validate_layout(layout)
    logical = _as_hwc(array, layout=layout_token)
    normalized_channels = _normalize_channels(channels)
    validated_colorspace = _validate_token(colorspace, axis="colorspace", accepted=_COLORSPACE_TOKENS)
    validated_gamma = _validate_token(gamma, axis="gamma", accepted=_GAMMA_TOKENS)
    if logical.shape[2] != len(normalized_channels):
        raise ValueError(
            _actionable_error(
                why="from_array channels metadata must match the input data channel axis",
                what=(
                    f"received data shape={logical.shape!r} with {logical.shape[2]} channels and "
                    f"channels={normalized_channels!r} with {len(normalized_channels)} labels"
                ),
                how=f"pass exactly {logical.shape[2]} labels, or provide data whose last axis has {len(normalized_channels)} channels",
            )
        )
    if np.dtype(logical.dtype) not in _ACCEPTED_DTYPES:
        raise ValueError(
            _actionable_error(
                why="from_array data dtype must be one of float32, float16, uint8, uint16, or uint32",
                what=f"received data dtype={logical.dtype!s} with shape={logical.shape!r}",
                how="convert input with data.astype(cp.float32) or another supported dtype before calling from_array",
            )
        )

    validated_bit_depth = None if bit_depth is None else _validate_bit_depth(bit_depth)
    affine_requested = any(value is not None for value in (scale, mean, std))
    if validated_bit_depth is not None:
        if affine_requested:
            raise ValueError(
                _actionable_error(
                    why="from_array bit_depth normalization cannot share a pass with affine scale, mean, or std",
                    what=f"received bit_depth={validated_bit_depth}, scale={scale!r}, mean={mean!r}, std={std!r}",
                    how=f"call from_array(..., bit_depth={validated_bit_depth}) without scale, mean, or std",
                )
            )
        container_dtype = _container_dtype(validated_bit_depth)
        if np.dtype(logical.dtype) != container_dtype:
            raise ValueError(
                _actionable_error(
                    why=f"from_array bit_depth={validated_bit_depth} requires {container_dtype.name} input storage",
                    what=f"received data dtype={logical.dtype!s}",
                    how=(
                        f"provide {container_dtype.name} input data, choose the bit_depth matching an"
                        " integer input container, or omit bit_depth for float input"
                    ),
                )
            )
        output_dtype = _resolve_dtype(dtype, default=np.dtype(np.float32))
        if output_dtype != np.dtype(np.float32):
            raise ValueError(
                _actionable_error(
                    why="from_array bit_depth normalization produces float32 Frame values",
                    what=f"received dtype={dtype!r}, resolved to {output_dtype.name!r}",
                    how="omit dtype or pass dtype='float32'",
                )
            )
    else:
        output_dtype = _resolve_dtype(dtype, default=np.dtype(logical.dtype))
    numeric_repacking = dtype is not None or affine_requested or validated_bit_depth is not None
    contiguity_repacking = not logical.flags.c_contiguous
    if copy is False and (numeric_repacking or contiguity_repacking):
        raise ValueError(
            _actionable_error(
                why="the declared layout, contiguity, dtype, or affine conversion requires a write",
                what="copy=False requires a zero-copy from_array input",
                how="use copy=None or copy=True, or provide contiguous HWC/NHWC data without numeric repacking",
            )
        )

    if numeric_repacking:
        output = cp.empty(logical.shape, dtype=output_dtype)
        channel_count = len(normalized_channels)
        affine_scale = _affine_values(scale, name="scale", channel_count=channel_count, default=1.0)
        affine_mean = _affine_values(mean, name="mean", channel_count=channel_count, default=0.0)
        affine_std = _affine_values(std, name="std", channel_count=channel_count, default=1.0)
        scalar_affine = channel_count <= _SCALAR_AFFINE_CHANNEL_LIMIT
        affine_arguments = _affine_kernel_arguments(
            affine_scale,
            affine_mean,
            affine_std,
            scalar_affine=scalar_affine,
        )
        pixel_count = int(logical.shape[0] * logical.shape[1])
        if pixel_count > 0:
            index_mode, strides = _from_array_index_mode(logical)
            block_count = (pixel_count + _FROM_ARRAY_THREADS_PER_BLOCK - 1) // _FROM_ARRAY_THREADS_PER_BLOCK
            _from_array_kernel(
                np.dtype(logical.dtype).name,
                output_dtype.name,
                channel_count,
                index_mode,
                scalar_affine,
                validated_bit_depth is not None,
            )(
                (block_count,),
                (_FROM_ARRAY_THREADS_PER_BLOCK,),
                (
                    logical,
                    output,
                    np.int64(pixel_count),
                    np.int64(logical.shape[1]),
                    np.int64(strides[0]),
                    np.int64(strides[1]),
                    np.int64(strides[2]),
                    _bit_depth_scale(validated_bit_depth) if validated_bit_depth is not None else np.float32(1.0),
                    *affine_arguments,
                ),
            )
    elif copy is True:
        output = logical.copy(order="C")
    elif contiguity_repacking:
        output = cp.ascontiguousarray(logical)
    else:
        output = logical
    return Frame(
        data=output,
        colorspace=validated_colorspace,
        gamma=validated_gamma,
        channels=normalized_channels,
        matrix=matrix,
    )
