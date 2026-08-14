"""Deterministic global and contrast-limited histogram equalization on the GPU."""

from __future__ import annotations

import math
from functools import lru_cache
from numbers import Real

import cupy as cp
import numpy as np

from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame, _new_frame, _validate_float32_frame

_THREADS_PER_BLOCK = 256

_HISTOGRAM_KERNEL_SOURCE = r"""
__device__ __forceinline__ long long pixtreme_histogram_bin(
    const float value,
    const double domain_minimum,
    const double domain_maximum,
    const long long bins
) {
    const double promoted = (double)value;
    if (promoted <= domain_minimum) {
        return 0;
    }
    if (promoted >= domain_maximum) {
        return bins - 1;
    }
    const double scaled = (promoted - domain_minimum) / (domain_maximum - domain_minimum) * (double)bins;
    const long long index = (long long)floor(scaled);
    return index < bins ? index : bins - 1;
}

__device__ __forceinline__ long long pixtreme_histogram_mirror_index(
    const long long index,
    const long long extent
) {
    if (extent <= 1) {
        return 0;
    }
    const long long period = 2 * extent - 2;
    const long long remainder = index % period;
    const long long reflected = remainder < 0 ? remainder + period : remainder;
    return reflected < extent ? reflected : period - reflected;
}

extern "C" __global__ void pixtreme_global_histogram(
    const float* __restrict__ source,
    unsigned long long* __restrict__ counts,
    const long long element_count,
    const long long channel_count,
    const long long bins,
    const double domain_minimum,
    const double domain_maximum
) {
    const long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= element_count) {
        return;
    }
    const long long channel = index % channel_count;
    const long long bin = pixtreme_histogram_bin(source[index], domain_minimum, domain_maximum, bins);
    atomicAdd(&counts[channel * bins + bin], 1ULL);
}

extern "C" __global__ void pixtreme_global_histogram_lookup(
    const float* __restrict__ source,
    const float* __restrict__ cdf,
    float* __restrict__ output,
    const long long element_count,
    const long long channel_count,
    const long long bins,
    const double domain_minimum,
    const double domain_maximum
) {
    const long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= element_count) {
        return;
    }
    const long long channel = index % channel_count;
    const long long bin = pixtreme_histogram_bin(source[index], domain_minimum, domain_maximum, bins);
    output[index] = cdf[channel * bins + bin];
}

extern "C" __global__ void pixtreme_clahe_histogram(
    const float* __restrict__ source,
    unsigned long long* __restrict__ counts,
    const long long padded_element_count,
    const long long width,
    const long long height,
    const long long channel_count,
    const long long tile_width,
    const long long tile_height,
    const long long tiles_x,
    const long long bins,
    const double domain_minimum,
    const double domain_maximum
) {
    const long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= padded_element_count) {
        return;
    }
    const long long channel = index % channel_count;
    const long long padded_pixel = index / channel_count;
    const long long padded_width = tiles_x * tile_width;
    const long long padded_x = padded_pixel % padded_width;
    const long long padded_y = padded_pixel / padded_width;
    const long long source_x = pixtreme_histogram_mirror_index(padded_x, width);
    const long long source_y = pixtreme_histogram_mirror_index(padded_y, height);
    const long long source_index = (source_y * width + source_x) * channel_count + channel;
    const long long tile_x = padded_x / tile_width;
    const long long tile_y = padded_y / tile_height;
    const long long histogram = (tile_y * tiles_x + tile_x) * channel_count + channel;
    const long long bin = pixtreme_histogram_bin(
        source[source_index], domain_minimum, domain_maximum, bins
    );
    atomicAdd(&counts[histogram * bins + bin], 1ULL);
}

extern "C" __global__ void pixtreme_clahe_waterfill_lut(
    const unsigned long long* __restrict__ counts,
    float* __restrict__ luts,
    const long long histogram_count,
    const long long bins,
    const long long tile_pixels,
    const double clip_limit
) {
    const long long histogram = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (histogram >= histogram_count) {
        return;
    }
    const long long base = histogram * bins;
    const double cap = clip_limit * (double)tile_pixels / (double)bins;
    double initial_sum = 0.0;
    for (long long bin = 0; bin < bins; ++bin) {
        initial_sum += fmin((double)counts[base + bin], cap);
    }

    double lambda = 0.0;
    if (initial_sum < (double)tile_pixels) {
        double lower = 0.0;
        double upper = cap;
        for (int iteration = 0; iteration < 80; ++iteration) {
            const double middle = lower + (upper - lower) * 0.5;
            double sum = 0.0;
            for (long long bin = 0; bin < bins; ++bin) {
                sum += fmin((double)counts[base + bin] + middle, cap);
            }
            if (sum < (double)tile_pixels) {
                lower = middle;
            } else {
                upper = middle;
            }
        }
        lambda = upper;
    }

    double cumulative = 0.0;
    for (long long bin = 0; bin < bins; ++bin) {
        const double filled = fmin((double)counts[base + bin] + lambda, cap);
        cumulative += filled;
        luts[base + bin] = bin == bins - 1 ? 1.0f : (float)(cumulative / (double)tile_pixels);
    }
}

extern "C" __global__ void pixtreme_clahe_lookup(
    const float* __restrict__ source,
    const float* __restrict__ luts,
    float* __restrict__ output,
    const long long element_count,
    const long long width,
    const long long height,
    const long long channel_count,
    const long long tile_width,
    const long long tile_height,
    const long long tiles_x,
    const long long tiles_y,
    const long long bins,
    const double domain_minimum,
    const double domain_maximum
) {
    const long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= element_count) {
        return;
    }
    const long long channel = index % channel_count;
    const long long pixel = index / channel_count;
    const long long x = pixel % width;
    const long long y = pixel / width;
    const long long bin = pixtreme_histogram_bin(source[index], domain_minimum, domain_maximum, bins);

    double u = ((double)x + 0.5) / (double)tile_width - 0.5;
    double v = ((double)y + 0.5) / (double)tile_height - 0.5;
    u = fmin(fmax(u, 0.0), (double)(tiles_x - 1));
    v = fmin(fmax(v, 0.0), (double)(tiles_y - 1));
    const long long tile_x0 = (long long)floor(u);
    const long long tile_y0 = (long long)floor(v);
    const long long tile_x1 = tile_x0 + 1 < tiles_x ? tile_x0 + 1 : tile_x0;
    const long long tile_y1 = tile_y0 + 1 < tiles_y ? tile_y0 + 1 : tile_y0;
    const double fraction_x = u - (double)tile_x0;
    const double fraction_y = v - (double)tile_y0;

    const long long top_left = ((tile_y0 * tiles_x + tile_x0) * channel_count + channel) * bins + bin;
    const long long top_right = ((tile_y0 * tiles_x + tile_x1) * channel_count + channel) * bins + bin;
    const long long bottom_left = ((tile_y1 * tiles_x + tile_x0) * channel_count + channel) * bins + bin;
    const long long bottom_right = ((tile_y1 * tiles_x + tile_x1) * channel_count + channel) * bins + bin;
    const double top = (1.0 - fraction_x) * (double)luts[top_left]
        + fraction_x * (double)luts[top_right];
    const double bottom = (1.0 - fraction_x) * (double)luts[bottom_left]
        + fraction_x * (double)luts[bottom_right];
    output[index] = (float)((1.0 - fraction_y) * top + fraction_y * bottom);
}
"""


@lru_cache(maxsize=1)
def _global_histogram_kernel() -> cp.RawKernel:
    return cp.RawKernel(_HISTOGRAM_KERNEL_SOURCE, "pixtreme_global_histogram")


@lru_cache(maxsize=1)
def _global_lookup_kernel() -> cp.RawKernel:
    return cp.RawKernel(_HISTOGRAM_KERNEL_SOURCE, "pixtreme_global_histogram_lookup")


@lru_cache(maxsize=1)
def _clahe_histogram_kernel() -> cp.RawKernel:
    return cp.RawKernel(_HISTOGRAM_KERNEL_SOURCE, "pixtreme_clahe_histogram")


@lru_cache(maxsize=1)
def _clahe_waterfill_kernel() -> cp.RawKernel:
    return cp.RawKernel(_HISTOGRAM_KERNEL_SOURCE, "pixtreme_clahe_waterfill_lut")


@lru_cache(maxsize=1)
def _clahe_lookup_kernel() -> cp.RawKernel:
    return cp.RawKernel(_HISTOGRAM_KERNEL_SOURCE, "pixtreme_clahe_lookup")


def _blocks(element_count: int) -> int:
    return (element_count + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK


def _validate_domain(value: object) -> tuple[float, float]:
    how = (
        "pass domain as a built-in tuple (minimum, maximum) of two real numbers that convert to finite float64 "
        "values, with minimum < maximum and a finite maximum - minimum width"
    )
    if type(value) is not tuple or len(value) != 2:
        raise ValueError(
            _actionable_error(
                why="domain must be a two-element built-in tuple",
                what=f"received domain={value!r}",
                how=how,
            )
        )
    minimum, maximum = value
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, Real)
        or isinstance(maximum, bool)
        or not isinstance(maximum, Real)
    ):
        raise ValueError(
            _actionable_error(
                why="domain elements must be real numbers other than bool",
                what=f"received domain={value!r}",
                how=how,
            )
        )
    try:
        resolved_minimum = float(minimum)
        resolved_maximum = float(maximum)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            _actionable_error(
                why="domain elements must convert to float64 values",
                what=f"received domain={value!r}",
                how=how,
            )
        ) from error
    if not math.isfinite(resolved_minimum) or not math.isfinite(resolved_maximum):
        raise ValueError(
            _actionable_error(
                why="domain elements must both be finite",
                what=f"received domain={value!r}",
                how=how,
            )
        )
    if resolved_minimum >= resolved_maximum:
        raise ValueError(
            _actionable_error(
                why="domain minimum must be strictly less than maximum",
                what=f"received domain={value!r}",
                how=how,
            )
        )
    if not math.isfinite(resolved_maximum - resolved_minimum):
        raise ValueError(
            _actionable_error(
                why="domain maximum - minimum width must be finite in float64",
                what=f"received domain={value!r}",
                how=how,
            )
        )
    return resolved_minimum, resolved_maximum


def _validate_bins(value: object) -> int:
    if type(value) is not int or not 2 <= value <= 65536:
        raise ValueError(
            _actionable_error(
                why="bins must be a built-in int in the inclusive range 2 through 65536",
                what=f"received bins={value!r}",
                how="pass a built-in int bins from 2 through 65536",
            )
        )
    return value


def _validate_clip_limit(value: object) -> float:
    how = "pass a real clip_limit that converts to a finite float64 value greater than or equal to 1.0"
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            _actionable_error(
                why="clip_limit must be a finite real number other than bool",
                what=f"received clip_limit={value!r}",
                how=how,
            )
        )
    try:
        resolved = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            _actionable_error(
                why="clip_limit must convert to a float64 value",
                what=f"received clip_limit={value!r}",
                how=how,
            )
        ) from error
    if not math.isfinite(resolved) or resolved < 1.0:
        raise ValueError(
            _actionable_error(
                why="clip_limit must be finite and greater than or equal to 1.0",
                what=f"received clip_limit={value!r}",
                how=how,
            )
        )
    return resolved


def _validate_tile_count(value: object, *, name: str, dimension: str, maximum: int) -> int:
    how = f"pass {name} as a positive built-in int no greater than Frame {dimension}={maximum}"
    if type(value) is not int:
        raise ValueError(
            _actionable_error(
                why=f"{name} must be a built-in int",
                what=f"received {name}={value!r}",
                how=how,
            )
        )
    if value < 1 or value > maximum:
        raise ValueError(
            _actionable_error(
                why=f"{name} must be positive and fit the input {dimension}",
                what=f"received {name}={value!r} for Frame {dimension}={maximum}",
                how=how,
            )
        )
    return value


def equalize_histogram(
    frame: Frame,
    *,
    domain: tuple[float, float] = (0.0, 1.0),
    bins: int = 1024,
) -> Frame:
    """Equalize each channel of a float32 Frame with its direct empirical CDF.

    The signature is ``equalize_histogram(frame, *, domain=(0.0, 1.0),
    bins=1024) -> Frame``. ``domain=(minimum, maximum)`` declares the histogram
    interval. Values are clamped to that interval and their bin is
    ``min(floor(normalized * bins), bins - 1)``. Processing is per channel: each
    channel builds its own direct empirical CDF from every pixel, with no
    CDF-min subtraction, input min-max inference, or channel combination.
    Output values lie in ``[0, 1]``, and a uniform channel maps to 1.0 rather
    than receiving a special case.

    The result is a private C-contiguous float32 Frame with shape and metadata
    preserved; the input remains unchanged. Convert non-float32 storage by its
    value meaning with ``px.values.cast_dtype``, ``px.values.recode_dtype``, or
    ``px.values.dequantize`` before calling this operation.
    """
    checked_frame = _validate_float32_frame(frame, operation="color.equalize_histogram")
    checked_domain = _validate_domain(domain)
    checked_bins = _validate_bins(bins)
    channel_count = len(checked_frame.channels)
    element_count = int(checked_frame.data.size)
    counts = cp.zeros((channel_count, checked_bins), dtype=cp.uint64)
    kernel_arguments = (
        np.int64(element_count),
        np.int64(channel_count),
        np.int64(checked_bins),
        np.float64(checked_domain[0]),
        np.float64(checked_domain[1]),
    )
    _global_histogram_kernel()(
        (_blocks(element_count),),
        (_THREADS_PER_BLOCK,),
        (checked_frame.data, counts, *kernel_arguments),
    )
    pixel_count = checked_frame.height * checked_frame.width
    cdf = cp.cumsum(counts, axis=1, dtype=cp.uint64).astype(cp.float64)
    cdf *= np.float64(1.0 / pixel_count)
    cdf = cdf.astype(cp.float32)
    output = cp.empty(checked_frame.shape, dtype=cp.float32)
    _global_lookup_kernel()(
        (_blocks(element_count),),
        (_THREADS_PER_BLOCK,),
        (checked_frame.data, cdf, output, *kernel_arguments),
    )
    return _new_frame(checked_frame, output)


def clahe(
    frame: Frame,
    *,
    clip_limit: float = 2.0,
    tiles_y: int = 8,
    tiles_x: int = 8,
    domain: tuple[float, float] = (0.0, 1.0),
    bins: int = 1024,
) -> Frame:
    """Apply deterministic contrast-limited adaptive histogram equalization.

    The signature is ``clahe(frame, *, clip_limit=2.0, tiles_y=8, tiles_x=8,
    domain=(0.0, 1.0), bins=1024) -> Frame``. ``tiles_y`` and ``tiles_x``
    declare the vertical and horizontal split counts. The ``domain`` tuple
    declares the histogram interval. Values are clamped to it and use
    ``min(floor(normalized * bins), bins - 1)``. Work is per channel without
    channel-label semantics. Fixed ceil-sized tiles receive bottom/right
    ``mirror`` padding with edge-excluding reflection. A tile bin is capped at
    ``cap = clip_limit * tile pixels / bins``; excess is restored by the unique
    uniform water-fill that preserves every count without exceeding the cap.
    Direct tile CDFs are looked up at tile-center positions, and bilinear
    interpolation clamps at the outer tile centers without wrap. Output values
    lie in ``[0, 1]`` without a final clip or renormalization.

    A choice where ``bins exceeds`` the padded tile pixel count is valid, but
    its sparse statistics and possible banding should be checked on the actual
    material. The result is a private C-contiguous float32 Frame with shape and
    metadata preserved; the input remains unchanged. Convert non-float32
    storage by its value meaning with ``px.values.cast_dtype``,
    ``px.values.recode_dtype``, or ``px.values.dequantize`` first.
    """
    checked_frame = _validate_float32_frame(frame, operation="color.clahe")
    checked_clip_limit = _validate_clip_limit(clip_limit)
    checked_tiles_y = _validate_tile_count(
        tiles_y,
        name="tiles_y",
        dimension="height",
        maximum=checked_frame.height,
    )
    checked_tiles_x = _validate_tile_count(
        tiles_x,
        name="tiles_x",
        dimension="width",
        maximum=checked_frame.width,
    )
    checked_domain = _validate_domain(domain)
    checked_bins = _validate_bins(bins)

    tile_height = (checked_frame.height + checked_tiles_y - 1) // checked_tiles_y
    tile_width = (checked_frame.width + checked_tiles_x - 1) // checked_tiles_x
    padded_height = checked_tiles_y * tile_height
    padded_width = checked_tiles_x * tile_width
    channel_count = len(checked_frame.channels)
    histogram_count = checked_tiles_y * checked_tiles_x * channel_count
    tile_pixels = tile_height * tile_width
    element_count = int(checked_frame.data.size)
    padded_element_count = padded_height * padded_width * channel_count

    counts = cp.zeros((histogram_count, checked_bins), dtype=cp.uint64)
    _clahe_histogram_kernel()(
        (_blocks(padded_element_count),),
        (_THREADS_PER_BLOCK,),
        (
            checked_frame.data,
            counts,
            np.int64(padded_element_count),
            np.int64(checked_frame.width),
            np.int64(checked_frame.height),
            np.int64(channel_count),
            np.int64(tile_width),
            np.int64(tile_height),
            np.int64(checked_tiles_x),
            np.int64(checked_bins),
            np.float64(checked_domain[0]),
            np.float64(checked_domain[1]),
        ),
    )
    luts = cp.empty((histogram_count, checked_bins), dtype=cp.float32)
    _clahe_waterfill_kernel()(
        (_blocks(histogram_count),),
        (_THREADS_PER_BLOCK,),
        (
            counts,
            luts,
            np.int64(histogram_count),
            np.int64(checked_bins),
            np.int64(tile_pixels),
            np.float64(checked_clip_limit),
        ),
    )
    output = cp.empty(checked_frame.shape, dtype=cp.float32)
    _clahe_lookup_kernel()(
        (_blocks(element_count),),
        (_THREADS_PER_BLOCK,),
        (
            checked_frame.data,
            luts,
            output,
            np.int64(element_count),
            np.int64(checked_frame.width),
            np.int64(checked_frame.height),
            np.int64(channel_count),
            np.int64(tile_width),
            np.int64(tile_height),
            np.int64(checked_tiles_x),
            np.int64(checked_tiles_y),
            np.int64(checked_bins),
            np.float64(checked_domain[0]),
            np.float64(checked_domain[1]),
        ),
    )
    return _new_frame(checked_frame, output)
