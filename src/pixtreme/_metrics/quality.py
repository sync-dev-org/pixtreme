"""GPU full-reference quality measurements."""

from __future__ import annotations

import math
from functools import lru_cache
from numbers import Real

import cupy as cp
import numpy as np

from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame
from pixtreme._core.value_domain import _float32_conversion_guidance
from pixtreme._metrics.common import _SSIM_KERNEL_SOURCE, _THREADS_PER_BLOCK, _block_count


@lru_cache(maxsize=1)
def _ssim_map_kernel() -> cp.RawKernel:
    return cp.RawKernel(_SSIM_KERNEL_SOURCE, "pixtreme_ssim_map")


@lru_cache(maxsize=1)
def _ssim_weights() -> cp.ndarray:
    offsets = np.arange(-5, 6, dtype=np.float64)
    yy, xx = np.meshgrid(offsets, offsets, indexing="ij")
    weights = np.exp(-(xx * xx + yy * yy) / (2.0 * 1.5**2))
    weights /= np.sum(weights, dtype=np.float64)
    return cp.asarray(weights, dtype=cp.float32)


def _validate_quality_frame(value: object, *, operation: str, argument: str) -> Frame:
    if not isinstance(value, Frame):
        raise ValueError(
            _actionable_error(
                why=f"{operation} {argument} must be a metadata-bearing float32 Frame",
                what=f"received {argument} as {type(value).__module__}.{type(value).__qualname__}",
                how=f"construct {argument} with px.io.from_array before calling px.metrics.{operation}",
            )
        )
    return value


def _validate_quality_dtypes(reference: Frame, candidate: Frame, *, operation: str) -> None:
    reference_dtype = np.dtype(reference.dtype)
    candidate_dtype = np.dtype(candidate.dtype)
    if reference_dtype == np.dtype(np.float32) and candidate_dtype == np.dtype(np.float32):
        return
    offending_dtype = reference_dtype if reference_dtype != np.dtype(np.float32) else candidate_dtype
    raise ValueError(
        _actionable_error(
            why=f"{operation} requires float32 Frame data for both inputs",
            what=f"received reference dtype {reference_dtype.name} and candidate dtype {candidate_dtype.name}",
            how=_float32_conversion_guidance(offending_dtype),
        )
    )


def _validate_quality_compatibility(reference: Frame, candidate: Frame, *, operation: str) -> None:
    comparisons = (
        ("height", reference.height, candidate.height),
        ("width", reference.width, candidate.width),
        ("channel count", len(reference.channels), len(candidate.channels)),
        ("channels", reference.channels, candidate.channels),
        ("colorspace", reference.colorspace, candidate.colorspace),
        ("gamma", reference.gamma, candidate.gamma),
        ("matrix", reference.matrix, candidate.matrix),
    )
    for name, reference_value, candidate_value in comparisons:
        if reference_value != candidate_value:
            raise ValueError(
                _actionable_error(
                    why=f"{operation} requires identical {name} on reference and candidate",
                    what=(f"received reference {name}={reference_value!r}, candidate {name}={candidate_value!r}"),
                    how=f"adapt the candidate explicitly so its {name} exactly matches the reference",
                )
            )


def _validate_quality_geometry(reference: Frame, candidate: Frame, *, operation: str) -> None:
    if operation != "psnr" and (reference.height < 11 or reference.width < 11):
        geometry = "height and width of at least 11"
        raise ValueError(
            _actionable_error(
                why=f"{operation} requires {geometry}",
                what=f"received reference shape {reference.shape!r} and candidate shape {candidate.shape!r}",
                how=f"provide matching Frames with {geometry}",
            )
        )


def _validate_data_range(value: object, *, operation: str) -> np.float32:
    guidance = "pass data_range as a positive finite real number"
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            _actionable_error(
                why=f"{operation} data_range must be a positive finite real number",
                what=f"received data_range={value!r}",
                how=guidance,
            )
        )
    try:
        resolved = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            _actionable_error(
                why=f"{operation} data_range must convert to finite float64",
                what=f"received data_range={value!r}",
                how=guidance,
            )
        ) from error
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(
            _actionable_error(
                why=f"{operation} data_range must be finite and greater than zero",
                what=f"received data_range={value!r}",
                how=guidance,
            )
        )
    with np.errstate(over="ignore"):
        return np.float32(resolved)


def _prepare_quality_inputs(
    reference: object,
    candidate: object,
    data_range: object,
    *,
    operation: str,
) -> tuple[Frame, Frame, np.float32]:
    checked_reference = _validate_quality_frame(reference, operation=operation, argument="reference")
    checked_candidate = _validate_quality_frame(candidate, operation=operation, argument="candidate")
    _validate_quality_dtypes(checked_reference, checked_candidate, operation=operation)
    _validate_quality_compatibility(checked_reference, checked_candidate, operation=operation)
    _validate_quality_geometry(checked_reference, checked_candidate, operation=operation)
    checked_data_range = _validate_data_range(data_range, operation=operation)
    return checked_reference, checked_candidate, checked_data_range


def psnr(reference: Frame, candidate: Frame, *, data_range: float = 1.0) -> cp.ndarray:
    """Return full-reference peak signal-to-noise ratio on the GPU.

    Exact signature: ``px.metrics.psnr(reference, candidate, *, data_range=1.0)``.
    Both inputs must be float32 Frame objects whose height, width, channels,
    colorspace, gamma, and matrix match literally. ``data_range`` is the
    caller-declared positive finite signal range and defaults to 1.0; it is not
    inferred from pixels, dtype, or metadata. Across all channels and spatial
    samples at once, ``MSE = sum((reference - candidate)**2) / (H * W * C)``
    and ``PSNR = 10 * log10(data_range**2 / MSE)``. Exact finite matches return
    ``+inf``. The float32 calculation does not clamp scene values or scan and
    replace non-finite inputs.

    The result is a private 0D float32 ``cupy.ndarray`` with new storage and no
    Frame metadata. It is GPU-resident and this call does not mutate either
    input. Choosing ``float(result)`` or ``result.item()`` performs explicit
    synchronization to obtain a host scalar. Convert non-float32 storage
    according to its meaning with ``px.values.cast_dtype``,
    ``px.values.recode_dtype``, or ``px.values.dequantize``. To measure luma,
    apply ``px.color.rgb_to_grayscale`` to both inputs first; use
    ``px.channel.shuffle`` for explicit channel selection or routing.
    """
    checked_reference, checked_candidate, checked_data_range = _prepare_quality_inputs(
        reference,
        candidate,
        data_range,
        operation="psnr",
    )
    difference = checked_reference.data - checked_candidate.data
    mse = cp.mean(difference * difference, dtype=cp.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        range_squared = np.float32(checked_data_range * checked_data_range)
    calculated = np.float32(10.0) * cp.log10(range_squared / mse)
    return cp.where(mse == np.float32(0.0), np.float32(np.inf), calculated).astype(cp.float32, copy=False)


def _ssim_map_checked(reference: Frame, candidate: Frame, *, data_range: np.float32) -> cp.ndarray:
    output_height = reference.height - 10
    output_width = reference.width - 10
    output = cp.empty((output_height, output_width), dtype=cp.float32)
    output_elements = output_height * output_width
    with np.errstate(over="ignore", invalid="ignore"):
        k1_range = np.float32(np.float32(0.01) * data_range)
        k2_range = np.float32(np.float32(0.03) * data_range)
        c1 = np.float32(k1_range * k1_range)
        c2 = np.float32(k2_range * k2_range)
    _ssim_map_kernel()(
        (_block_count(output_elements),),
        (_THREADS_PER_BLOCK,),
        (
            reference.data,
            candidate.data,
            _ssim_weights(),
            output,
            np.int64(reference.width),
            np.int64(reference.height),
            np.int64(len(reference.channels)),
            np.float32(c1),
            np.float32(c2),
        ),
    )
    return output


def ssim(reference: Frame, candidate: Frame, *, data_range: float = 1.0) -> cp.ndarray:
    """Return the spatial mean of the valid full-reference SSIM map on the GPU.

    Exact signature: ``px.metrics.ssim(reference, candidate, *, data_range=1.0)``.
    Both inputs must be float32 Frame objects whose height, width, channels,
    colorspace, gamma, and matrix match literally. ``data_range`` is the
    caller-declared positive finite signal range and defaults to 1.0; pixels
    are never used to infer it. The metric uses a normalized 11x11 Gaussian
    window with ``sigma=1.5``, weighted population variance and covariance,
    ``C1 = (0.01 * data_range)**2``, and ``C2 = (0.03 * data_range)**2``.
    It evaluates each channel independently, takes the channel mean at each
    valid position, then takes the float32 spatial mean over all channels'
    local contributions. It does not clamp scene values or replace non-finite
    inputs.

    The result is a private 0D float32 ``cupy.ndarray`` with new storage and no
    Frame metadata. It is GPU-resident and this call does not mutate either
    input. Choosing ``float(result)`` or ``result.item()`` performs explicit
    synchronization to obtain a host scalar. Convert non-float32 storage
    according to its meaning with ``px.values.cast_dtype``,
    ``px.values.recode_dtype``, or ``px.values.dequantize``. To measure luma,
    apply ``px.color.rgb_to_grayscale`` to both inputs first; use
    ``px.channel.shuffle`` for explicit channel selection or routing.
    """
    checked_reference, checked_candidate, checked_data_range = _prepare_quality_inputs(
        reference,
        candidate,
        data_range,
        operation="ssim",
    )
    local_map = _ssim_map_checked(checked_reference, checked_candidate, data_range=checked_data_range)
    return cp.mean(local_map, dtype=cp.float32)


def ssim_map(reference: Frame, candidate: Frame, *, data_range: float = 1.0) -> cp.ndarray:
    """Return the valid full-reference SSIM response map on the GPU.

    Exact signature: ``px.metrics.ssim_map(reference, candidate, *, data_range=1.0)``.
    Both inputs must be float32 Frame objects whose height, width, channels,
    colorspace, gamma, and matrix match literally. ``data_range`` is the
    caller-declared positive finite signal range and defaults to 1.0; pixels
    are never used to infer it. Each valid 11x11 Gaussian window uses
    ``sigma=1.5``, weighted population variance and covariance,
    ``C1 = (0.01 * data_range)**2``, and ``C2 = (0.03 * data_range)**2``.
    The expression is evaluated independently for all channels, followed by a
    channel mean. There is no padding: HWC input produces the 2D shape
    ``(H - 10, W - 10)``. The calculation does not clamp scene values or
    replace non-finite inputs.

    The result is a private C-contiguous 2D float32 ``cupy.ndarray`` with new
    storage and no Frame metadata, and this call does not mutate either input.
    Convert non-float32 storage according to its meaning with
    ``px.values.cast_dtype``, ``px.values.recode_dtype``, or
    ``px.values.dequantize``. For visualization, add a length-one channel
    dimension and call ``px.io.from_array`` with explicit metadata. To measure
    luma, apply ``px.color.rgb_to_grayscale`` to both inputs first; use
    ``px.channel.shuffle`` for explicit channel selection or routing.
    """
    checked_reference, checked_candidate, checked_data_range = _prepare_quality_inputs(
        reference,
        candidate,
        data_range,
        operation="ssim_map",
    )
    return _ssim_map_checked(checked_reference, checked_candidate, data_range=checked_data_range)
