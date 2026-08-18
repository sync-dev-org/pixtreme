"""GPU feature measurements that return raw two-dimensional response arrays."""

from __future__ import annotations

import math
import warnings
from functools import lru_cache
from numbers import Real

import cupy as cp
import numpy as np

from pixtreme._core.border import _border_argument, _resolve_border
from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame, _validate_float32_frame
from pixtreme._core.value_domain import _float32_conversion_guidance
from pixtreme._core.vocabulary import _TEMPLATE_MATCHING_METHOD_TOKENS
from pixtreme._feature.common import _ANALYZE_KERNEL_SOURCE, _THREADS_PER_BLOCK, _block_count

_METHOD_TOKENS = _TEMPLATE_MATCHING_METHOD_TOKENS


_DIRECT_MATCH_OPERATION_LIMIT = 8_000_000


@lru_cache(maxsize=1)
def _harris_tensor_kernel() -> cp.RawKernel:
    return cp.RawKernel(_ANALYZE_KERNEL_SOURCE, "pixtreme_harris_tensor")


@lru_cache(maxsize=1)
def _harris_response_kernel() -> cp.RawKernel:
    return cp.RawKernel(_ANALYZE_KERNEL_SOURCE, "pixtreme_harris_response")


@lru_cache(maxsize=1)
def _match_template_kernel() -> cp.RawKernel:
    return cp.RawKernel(_ANALYZE_KERNEL_SOURCE, "pixtreme_match_template")


@lru_cache(maxsize=1)
def _match_template_fft_response_kernel() -> cp.RawKernel:
    return cp.RawKernel(_ANALYZE_KERNEL_SOURCE, "pixtreme_match_template_fft_response")


def _validate_block_size(value: object) -> int:
    if type(value) is not int or value < 1 or value % 2 == 0:
        raise ValueError(
            _actionable_error(
                why="corner_harris block_size must be a centered positive odd built-in int",
                what=f"received block_size={value!r}",
                how="pass an odd built-in int of at least 1; the window may exceed the image dimensions",
            )
        )
    return value


def _validate_k(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            _actionable_error(
                why="corner_harris k must be a real Harris coefficient",
                what=f"received k={value!r}",
                how="pass a real number in the open interval (0.0, 0.25)",
            )
        )
    try:
        resolved = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            _actionable_error(
                why="corner_harris k must convert to finite float64",
                what=f"received k={value!r}",
                how="pass a real number in the open interval (0.0, 0.25)",
            )
        ) from error
    if not math.isfinite(resolved) or not 0.0 < resolved < 0.25:
        raise ValueError(
            _actionable_error(
                why="corner_harris k must be finite and strictly between 0.0 and 0.25",
                what=f"received k={value!r}",
                how="pass a real number in the open interval (0.0, 0.25)",
            )
        )
    return resolved


def _validate_frame_argument(value: object, *, argument: str) -> Frame:
    if not isinstance(value, Frame):
        raise ValueError(
            _actionable_error(
                why=f"match_template {argument} must be a metadata-bearing Frame",
                what=f"received {argument} as {type(value).__module__}.{type(value).__qualname__}",
                how=f"construct {argument} with px.io.from_array before calling px.feature.match_template",
            )
        )
    return value


def _validate_match_dtypes(frame: Frame, template: Frame) -> None:
    frame_dtype = np.dtype(frame.dtype)
    template_dtype = np.dtype(template.dtype)
    if frame_dtype == np.dtype(np.float32) and template_dtype == np.dtype(np.float32):
        return
    offending_dtype = frame_dtype if frame_dtype != np.dtype(np.float32) else template_dtype
    raise ValueError(
        _actionable_error(
            why="match_template requires float32 Frame data for both inputs",
            what=f"received frame dtype {frame_dtype.name} and template dtype {template_dtype.name}",
            how=_float32_conversion_guidance(offending_dtype),
        )
    )


def _validate_match_compatibility(frame: Frame, template: Frame) -> None:
    comparisons = (
        ("channel count", len(frame.channels), len(template.channels)),
        ("channels", frame.channels, template.channels),
        ("colorspace", frame.colorspace, template.colorspace),
        ("gamma", frame.gamma, template.gamma),
        ("matrix", frame.matrix, template.matrix),
    )
    for name, frame_value, template_value in comparisons:
        if frame_value != template_value:
            raise ValueError(
                _actionable_error(
                    why=f"match_template requires identical {name} on frame and template",
                    what=f"received frame {name}={frame_value!r}, template {name}={template_value!r}",
                    how=f"provide a template whose {name} exactly matches the frame; adapt it explicitly first",
                )
            )


def _validate_match_geometry(frame: Frame, template: Frame) -> None:
    if template.height > frame.height or template.width > frame.width:
        raise ValueError(
            _actionable_error(
                why="match_template requires valid template placement inside the frame",
                what=f"received frame shape {frame.shape!r} and template shape {template.shape!r}",
                how="use a template whose height and width are no larger than the frame",
            )
        )


def _validate_method(value: object) -> str:
    if not isinstance(value, str) or value not in _METHOD_TOKENS:
        raise ValueError(
            _actionable_error(
                why="method is a closed, case-sensitive template matching token axis",
                what=f"received method={value!r}",
                how=f"pass one of {_METHOD_TOKENS!r}",
            )
        )
    return value


def _window_sums(
    source: cp.ndarray,
    *,
    height: int,
    width: int,
    dtype: type[np.float32] | type[np.int32] | type[np.int64],
) -> cp.ndarray:
    integral = cp.cumsum(cp.cumsum(source, axis=0, dtype=dtype), axis=1, dtype=dtype)
    padded = cp.zeros((source.shape[0] + 1, source.shape[1] + 1, source.shape[2]), dtype=dtype)
    padded[1:, 1:, :] = integral
    return (
        padded[height:, width:, :]
        - padded[:-height, width:, :]
        - padded[height:, :-width, :]
        + padded[:-height, :-width, :]
    )


def _floating_window_sums(source: cp.ndarray, *, height: int, width: int) -> cp.ndarray:
    return _window_sums(source, height=height, width=width, dtype=cp.float32)


def _change_window_sums(source: cp.ndarray, *, height: int, width: int) -> cp.ndarray:
    dtype = cp.int32 if height * width <= np.iinfo(np.int32).max else cp.int64
    return _window_sums(source, height=height, width=width, dtype=dtype)


def _constant_window_mask(source: cp.ndarray, *, height: int, width: int) -> cp.ndarray:
    """Identify exact per-channel constants without a numeric threshold."""
    constant = cp.ones((source.shape[0] - height + 1, source.shape[1] - width + 1), dtype=cp.bool_)
    if height > 1:
        vertical_changes = cp.any(source[1:, :, :] != source[:-1, :, :], axis=2, keepdims=True)
        constant &= _change_window_sums(vertical_changes, height=height - 1, width=width)[..., 0] == 0
    if width > 1:
        horizontal_changes = cp.any(source[:, 1:, :] != source[:, :-1, :], axis=2, keepdims=True)
        constant &= _change_window_sums(horizontal_changes, height=height, width=width - 1)[..., 0] == 0

    return constant


def _normalized_response(numerator: cp.ndarray, denominator_squared: cp.ndarray, *, sqdiff: bool) -> cp.ndarray:
    positive = denominator_squared > np.float32(0.0)
    quotient = numerator / cp.sqrt(denominator_squared)
    if sqdiff:
        zero_denominator = cp.where(numerator > np.float32(0.0), np.float32(np.inf), np.float32(0.0))
        return cp.where(positive, quotient, zero_denominator)
    return cp.where(positive, quotient, np.float32(0.0))


def _fused_match_template_fft_response(
    correlation: cp.ndarray,
    window_sums: cp.ndarray,
    squared_window_sums: cp.ndarray,
    template_sums: cp.ndarray,
    template_energy: cp.ndarray,
    zero_variance: cp.ndarray,
    *,
    spatial_count: np.float32,
    method: str,
) -> cp.ndarray:
    if window_sums.shape[2] > 3:
        if method == "ccorr_normed":
            source_energy = cp.sum(squared_window_sums, axis=2, dtype=cp.float32)
            denominator_squared = source_energy * template_energy
            return cp.ascontiguousarray(
                _normalized_response(correlation, denominator_squared, sqdiff=False),
                dtype=cp.float32,
            )
        numerator = correlation - cp.sum(
            window_sums * template_sums[None, None, :] / spatial_count,
            axis=2,
            dtype=cp.float32,
        )
        if method == "ccoeff":
            return cp.ascontiguousarray(cp.where(zero_variance, np.float32(0.0), numerator), dtype=cp.float32)
        source_energy = cp.sum(squared_window_sums, axis=2, dtype=cp.float32)
        centered_source_energy = source_energy - cp.sum(
            window_sums * window_sums / spatial_count,
            axis=2,
            dtype=cp.float32,
        )
        centered_template_energy = template_energy - cp.sum(
            template_sums * template_sums / spatial_count,
            dtype=cp.float32,
        )
        denominator_squared = cp.where(
            zero_variance,
            np.float32(0.0),
            centered_source_energy * centered_template_energy,
        )
        return cp.ascontiguousarray(
            _normalized_response(numerator, denominator_squared, sqdiff=False),
            dtype=cp.float32,
        )

    centered_template_energy = template_energy
    if method == "ccoeff_normed":
        centered_template_energy = template_energy - cp.sum(
            template_sums * template_sums / spatial_count,
            dtype=cp.float32,
        )
    output = cp.empty(correlation.shape, dtype=cp.float32)
    _match_template_fft_response_kernel()(
        (_block_count(output.size),),
        (_THREADS_PER_BLOCK,),
        (
            correlation,
            window_sums,
            squared_window_sums,
            template_sums,
            template_energy,
            centered_template_energy,
            zero_variance,
            output,
            np.int64(output.size),
            np.int64(window_sums.shape[2]),
            spatial_count,
            np.int32(_METHOD_TOKENS.index(method)),
        ),
    )
    return output


def _match_template_fft(frame: Frame, template: Frame, *, method: str) -> cp.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from cupyx.scipy.signal import fftconvolve

    template_height = template.height
    template_width = template.width
    spatial_count = np.float32(template_height * template_width)
    correlation_channels = fftconvolve(
        frame.data,
        template.data[::-1, ::-1, :],
        mode="valid",
        axes=(0, 1),
    )
    correlation = cp.sum(correlation_channels, axis=2, dtype=cp.float32)
    if method == "ccorr":
        return cp.ascontiguousarray(correlation, dtype=cp.float32)

    if method == "ccorr_normed":
        squared_window_sums = _floating_window_sums(
            frame.data * frame.data,
            height=template_height,
            width=template_width,
        )
        template_energy = cp.sum(template.data * template.data, dtype=cp.float32)
        return _fused_match_template_fft_response(
            correlation,
            squared_window_sums,
            squared_window_sums,
            squared_window_sums,
            template_energy,
            correlation,
            spatial_count=spatial_count,
            method=method,
        )

    window_sums = _floating_window_sums(frame.data, height=template_height, width=template_width)
    template_sums = cp.sum(template.data, axis=(0, 1), dtype=cp.float32)
    zero_variance = _constant_window_mask(
        frame.data,
        height=template_height,
        width=template_width,
    ) | cp.all(template.data == template.data[0, 0, :])
    if method == "ccoeff":
        return _fused_match_template_fft_response(
            correlation,
            window_sums,
            window_sums,
            template_sums,
            template_sums,
            zero_variance,
            spatial_count=spatial_count,
            method=method,
        )
    squared_window_sums = _floating_window_sums(
        frame.data * frame.data,
        height=template_height,
        width=template_width,
    )
    template_energy = cp.sum(template.data * template.data, dtype=cp.float32)
    return _fused_match_template_fft_response(
        correlation,
        window_sums,
        squared_window_sums,
        template_sums,
        template_energy,
        zero_variance,
        spatial_count=spatial_count,
        method=method,
    )


def corner_harris(
    frame: Frame,
    *,
    block_size: int = 3,
    k: float = 0.04,
    border: str = "mirror",
    border_value: float | None = None,
) -> cp.ndarray:
    """Return the color-structure Harris response of a float32 Frame.

    The fixed non-normalized 3x3 Sobel pair is evaluated for all channels.
    Per-channel products are combined into one color tensor before a centered,
    non-normalized ``block_size`` box sum. The final expression is
    ``A * D - B**2 - k * (A + D)**2``. At output ``(y, x)`` it measures the
    input pixel at ``(y, x)``. ``mirror`` is the default border; ``replicate``,
    ``wrap``, and ``constant`` are also accepted, and the same extended input
    plane defines both Sobel and box sum samples.

    The result is a private 2D C-contiguous float32 ``cupy.ndarray``.
    It has new storage and no Frame metadata. This operation does not mutate its input,
    does not clamp scene values, and neither adds nor preserves a channel
    dimension. Convert non-float32 storage according to its meaning with
    ``px.values.cast_dtype``, ``px.values.recode_dtype``, or
    ``px.values.dequantize``. To use the response as an image, add a length-one
    channel dimension and call ``px.io.from_array`` with explicit metadata.
    """
    checked_frame = _validate_float32_frame(frame, operation="feature.corner_harris")
    checked_block_size = _validate_block_size(block_size)
    checked_k = _validate_k(k)
    checked_border, checked_border_value = _resolve_border(border, border_value)
    radius = checked_block_size // 2
    extended_height = checked_frame.height + 2 * radius
    extended_width = checked_frame.width + 2 * radius
    tensor = cp.empty((extended_height, extended_width, 3), dtype=cp.float32)
    tensor_elements = extended_height * extended_width
    _harris_tensor_kernel()(
        (_block_count(tensor_elements),),
        (_THREADS_PER_BLOCK,),
        (
            checked_frame.data,
            tensor,
            np.int64(checked_frame.width),
            np.int64(checked_frame.height),
            np.int64(len(checked_frame.channels)),
            np.int64(radius),
            _border_argument(checked_border),
            np.float32(checked_border_value),
        ),
    )
    output = cp.empty((checked_frame.height, checked_frame.width), dtype=cp.float32)
    output_elements = checked_frame.height * checked_frame.width
    _harris_response_kernel()(
        (_block_count(output_elements),),
        (_THREADS_PER_BLOCK,),
        (
            tensor,
            output,
            np.int64(checked_frame.width),
            np.int64(checked_frame.height),
            np.int64(radius),
            np.float32(checked_k),
        ),
    )
    return output


def match_template(frame: Frame, template: Frame, *, method: str = "ccoeff_normed") -> cp.ndarray:
    """Return a valid 2D template response map for two compatible float32 Frames.

    Output ``(y, x)`` compares the template with the frame window whose top-left
    pixel is ``(y, x)``; no padding is used. Every metric combines all channels.
    ``sqdiff`` and ``sqdiff_normed`` are smaller-is-better; ``ccorr``,
    ``ccorr_normed``, ``ccoeff``, and the default ``ccoeff_normed`` are
    larger-is-better. Coefficient means are computed per channel. The two
    Frames must have identical dtype, channels, colorspace, gamma, and matrix.
    A zero normalized denominator returns zero, except nonzero
    ``sqdiff_normed`` returns ``+inf``.

    The result is a private 2D C-contiguous float32 ``cupy.ndarray``.
    It has new storage and no Frame metadata. This operation does not mutate either input,
    does not clamp scene values, and neither adds nor preserves a channel
    dimension. Convert non-float32 storage according to its meaning with
    ``px.values.cast_dtype``, ``px.values.recode_dtype``, or
    ``px.values.dequantize``. To use a response as an image, add a length-one
    channel dimension and call ``px.io.from_array`` with explicit metadata.
    """
    checked_frame = _validate_frame_argument(frame, argument="frame")
    checked_template = _validate_frame_argument(template, argument="template")
    _validate_match_dtypes(checked_frame, checked_template)
    _validate_match_compatibility(checked_frame, checked_template)
    _validate_match_geometry(checked_frame, checked_template)
    checked_method = _validate_method(method)
    output_height = checked_frame.height - checked_template.height + 1
    output_width = checked_frame.width - checked_template.width + 1
    direct_operation_count = (
        output_height * output_width * checked_template.height * checked_template.width * len(checked_frame.channels)
    )
    if checked_method.startswith("cc") and direct_operation_count > _DIRECT_MATCH_OPERATION_LIMIT:
        return _match_template_fft(checked_frame, checked_template, method=checked_method)
    output = cp.empty((output_height, output_width), dtype=cp.float32)
    output_elements = output_height * output_width
    _match_template_kernel()(
        (_block_count(output_elements),),
        (_THREADS_PER_BLOCK,),
        (
            checked_frame.data,
            checked_template.data,
            output,
            np.int64(checked_frame.width),
            np.int64(checked_frame.height),
            np.int64(len(checked_frame.channels)),
            np.int64(checked_template.width),
            np.int64(checked_template.height),
            np.int32(_METHOD_TOKENS.index(checked_method)),
        ),
    )
    return output
