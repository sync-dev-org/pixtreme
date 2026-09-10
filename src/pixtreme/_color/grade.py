"""Per-channel Lift / Gamma / Gain grading."""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import lru_cache
from numbers import Real

import cupy as cp
import numpy as np

from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame, _new_frame, _validate_float32_frame

_NEUTRAL_LIFT = np.float32(0.0)
_NEUTRAL_GAMMA = np.float32(1.0)
_NEUTRAL_GAIN = np.float32(1.0)

_GRADE_KERNEL_SOURCE = r"""
extern "C" __global__ void pixtreme_grade(
    const unsigned int* __restrict__ source,
    unsigned int* __restrict__ destination,
    const float* __restrict__ parameters,
    const long long element_count,
    const int channel_count
) {
    const long long element = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= element_count) {
        return;
    }

    const int channel = (int)(element % channel_count);
    const float lift = parameters[3 * channel];
    const float gamma = parameters[3 * channel + 1];
    const float gain = parameters[3 * channel + 2];
    if (lift == 0.0f && gamma == 1.0f && gain == 1.0f) {
        destination[element] = source[element];
        return;
    }

    const float value = __uint_as_float(source[element]);
    const float z = fmaf(lift, 1.0f - value, gain * value);
    const float output = copysignf(powf(fabsf(z), 1.0f / gamma), z);
    destination[element] = __float_as_uint(output);
}
"""


def _error(*, why: str, what: str, how: str) -> ValueError:
    return ValueError(_actionable_error(why=why, what=what, how=how))


def _parameter_value(value: object, *, parameter: str, key: str | None) -> np.float32:
    location = parameter if key is None else f"{parameter}[{key!r}]"
    if isinstance(value, bool) or not isinstance(value, Real):
        raise _error(
            why=f"{location} must be a real number other than bool",
            what=f"received {location}={value!r}",
            how=f"pass {location} as a finite real value representable by float32",
        )
    try:
        host_value = float(value)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            resolved = np.float32(host_value)
    except (OverflowError, TypeError, ValueError) as error:
        raise _error(
            why=f"{location} must convert to binary32 before pixel processing",
            what=f"received {location}={value!r}",
            how=f"pass {location} as a finite real value representable by float32",
        ) from error
    if not math.isfinite(host_value) or not bool(np.isfinite(resolved)):
        raise _error(
            why=f"{location} must remain finite after conversion to binary32",
            what=f"received {location}={value!r}, converted={resolved!r}",
            how=f"pass {location} within the finite float32 range",
        )
    if parameter == "gamma" and not bool(resolved > np.float32(0.0)):
        raise _error(
            why=f"{location} must remain greater than zero after conversion to binary32",
            what=f"received {location}={value!r}, converted={resolved!r}",
            how=f"pass {location} as a positive finite value that does not underflow in float32",
        )
    return resolved


def _resolve_parameter(
    value: float | Mapping[str, float],
    *,
    parameter: str,
    channels: tuple[str, ...],
    neutral: np.float32,
) -> tuple[np.float32, ...]:
    if isinstance(value, Mapping):
        resolved: dict[str, np.float32] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise _error(
                    why=f"{parameter} mapping keys must be non-empty strings",
                    what=f"received {parameter} key={key!r} in {value!r}",
                    how=f"use exact case-sensitive labels from the input Frame channels {channels!r}",
                )
            if key not in channels:
                raise _error(
                    why=f"{parameter} mapping keys must name channels present in the input Frame",
                    what=f"received {parameter} key={key!r}; Frame channels={channels!r}",
                    how=f"use exact case-sensitive labels from the input Frame channels {channels!r}",
                )
            resolved[key] = _parameter_value(item, parameter=parameter, key=key)
        return tuple(resolved.get(label, neutral) for label in channels)
    scalar = _parameter_value(value, parameter=parameter, key=None)
    return (scalar,) * len(channels)


@lru_cache(maxsize=1)
def _grade_kernel() -> cp.RawKernel:
    return cp.RawKernel(_GRADE_KERNEL_SOURCE, "pixtreme_grade")


def grade(
    frame: Frame,
    *,
    lift: float | Mapping[str, float] = 0.0,
    gamma: float | Mapping[str, float] = 1.0,
    gain: float | Mapping[str, float] = 1.0,
) -> Frame:
    """Apply one label-resolved Lift / Gamma / Gain curve to stored samples.

    For binary32 values, ``z = gain * x + lift * (1 - x)`` and output is
    sign-preserving ``abs(z) ** (1 / gamma)`` with no clipping. Parameter
    ``gamma`` is the exponent denominator; it does not read or change the
    transfer metadata in ``frame.gamma``. Neutral values are lift 0, gamma 1,
    and gain 1. A scalar broadcasts to every storage channel, while a Mapping
    uses exact case-sensitive labels and leaves each missing parameter neutral.

    Channels are an open set, so scalar broadcast also grades A, Z, Cb, Cr, H,
    and custom labels; callers use a Mapping when 0 is black and 1 is white only
    for selected intensity channels. This differs from Nuke's default RGB
    selection and Blender / OCIO alpha preservation. The affine-power core
    matches Nuke only on its default-control 0-to-1 power interval, ASC CDL by
    slope=gain-lift, offset=lift, power=1/gamma (with a narrower valid ASC CDL
    subset), and classic LGG when lift is converted by gain. Extended-range
    clamp rules differ, and Resolve Primaries compatibility is not claimed.

    The result always owns new C-contiguous float32 storage and preserves all
    Frame metadata. A channel whose three binary32 parameters are neutral is
    copied bit-for-bit, including signed zero, NaN payload, and infinity.
    """
    checked_frame = _validate_float32_frame(frame, operation="color.grade")
    resolved_lift = _resolve_parameter(
        lift,
        parameter="lift",
        channels=checked_frame.channels,
        neutral=_NEUTRAL_LIFT,
    )
    resolved_gamma = _resolve_parameter(
        gamma,
        parameter="gamma",
        channels=checked_frame.channels,
        neutral=_NEUTRAL_GAMMA,
    )
    resolved_gain = _resolve_parameter(
        gain,
        parameter="gain",
        channels=checked_frame.channels,
        neutral=_NEUTRAL_GAIN,
    )
    neutral = all(
        channel_lift == _NEUTRAL_LIFT and channel_gamma == _NEUTRAL_GAMMA and channel_gain == _NEUTRAL_GAIN
        for channel_lift, channel_gamma, channel_gain in zip(
            resolved_lift,
            resolved_gamma,
            resolved_gain,
            strict=True,
        )
    )

    try:
        if neutral:
            output = checked_frame.data.copy()
        else:
            host_parameters = np.empty((len(checked_frame.channels), 3), dtype=np.float32)
            host_parameters[:, 0] = resolved_lift
            host_parameters[:, 1] = resolved_gamma
            host_parameters[:, 2] = resolved_gain
            with cp.cuda.Device(checked_frame.data.device.id):
                device_parameters = cp.asarray(host_parameters)
                output = cp.empty_like(checked_frame.data)
                element_count = checked_frame.data.size
                threads = 256
                _grade_kernel()(
                    ((element_count + threads - 1) // threads,),
                    (threads,),
                    (
                        checked_frame.data,
                        output,
                        device_parameters,
                        np.int64(element_count),
                        np.int32(len(checked_frame.channels)),
                    ),
                )
    except Exception as error:
        raise _error(
            why="color.grade could not execute its GPU pixel pass",
            what=f"backend raised {type(error).__module__}.{type(error).__qualname__}: {error}",
            how="verify the CUDA runtime and retry with a valid float32 Frame on an available NVIDIA GPU",
        ) from error
    return _new_frame(checked_frame, output)
