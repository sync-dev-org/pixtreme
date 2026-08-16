"""GPU-native deterministic procedural noise generators."""

from __future__ import annotations

from functools import lru_cache

import cupy as cp
import numpy as np

from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame
from pixtreme._core.validation import _bounded_real, _finite_real, _positive_real, _strict_bool
from pixtreme._generate.patterns import _dimension, _metadata

_NOISE_BLOCK = (16, 16)
_UINT32_MODULUS = 1 << 32

_NOISE_KERNEL_SOURCE = r"""
__device__ uint4 pixtreme_pcg4d(uint4 value) {
    value.x = value.x * 1664525u + 1013904223u;
    value.y = value.y * 1664525u + 1013904223u;
    value.z = value.z * 1664525u + 1013904223u;
    value.w = value.w * 1664525u + 1013904223u;
    value.x += value.y * value.w;
    value.y += value.z * value.x;
    value.z += value.x * value.y;
    value.w += value.y * value.z;
    value.x ^= value.x >> 16u;
    value.y ^= value.y >> 16u;
    value.z ^= value.z >> 16u;
    value.w ^= value.w >> 16u;
    value.x += value.y * value.w;
    value.y += value.z * value.x;
    value.z += value.x * value.y;
    value.w += value.y * value.z;
    return value;
}

__device__ unsigned int pixtreme_stream(
    const unsigned int seed,
    const unsigned long long octave,
    const unsigned int channel
) {
    return seed ^ (0x9e3779b9u * (unsigned int)octave) ^ (0x85ebca6bu * channel);
}

__device__ unsigned int pixtreme_wrapped_floor(const double floored) {
    double wrapped = fmod(floored, 4294967296.0);
    if (wrapped < 0.0) {
        wrapped += 4294967296.0;
    }
    return (unsigned int)wrapped;
}

__device__ float pixtreme_fade(const float value) {
    return value * value * value * (value * (value * 6.0f - 15.0f) + 10.0f);
}

__device__ float pixtreme_lerp(const float first, const float second, const float amount) {
    return first + amount * (second - first);
}

__device__ float3 pixtreme_gradient(
    const unsigned int x,
    const unsigned int y,
    const unsigned int z,
    const unsigned int stream
) {
    const uint4 hashed = pixtreme_pcg4d(make_uint4(x, y, z, stream));
    float3 gradient = make_float3(
        (float)((double)hashed.x * (2.0 / 4294967295.0) - 1.0),
        (float)((double)hashed.y * (2.0 / 4294967295.0) - 1.0),
        (float)((double)hashed.z * (2.0 / 4294967295.0) - 1.0)
    );
    const float norm = sqrtf(
        gradient.x * gradient.x +
        gradient.y * gradient.y +
        gradient.z * gradient.z
    );
    if (norm == 0.0f) {
        return make_float3(1.0f, 0.0f, 0.0f);
    }
    gradient.x /= norm;
    gradient.y /= norm;
    gradient.z /= norm;
    return gradient;
}

__device__ float pixtreme_gradient_dot(
    const unsigned int x,
    const unsigned int y,
    const unsigned int z,
    const unsigned int stream,
    const float offset_x,
    const float offset_y,
    const float offset_z
) {
    const float3 gradient = pixtreme_gradient(x, y, z, stream);
    return (
        gradient.x * offset_x +
        gradient.y * offset_y +
        gradient.z * offset_z
    );
}

__device__ double pixtreme_representable_lattice_coordinate(const double value) {
    if (!isfinite(value) || fabs(value) >= 9007199254740992.0) {
        return 0.0;
    }
    return value;
}

__device__ float pixtreme_gradient_noise(
    const double x,
    const double y,
    const double z,
    const unsigned int stream
) {
    const double finite_x = pixtreme_representable_lattice_coordinate(x);
    const double finite_y = pixtreme_representable_lattice_coordinate(y);
    const double floor_x = floor(finite_x);
    const double floor_y = floor(finite_y);
    const double floor_z = floor(z);
    const unsigned int ix = pixtreme_wrapped_floor(floor_x);
    const unsigned int iy = pixtreme_wrapped_floor(floor_y);
    const unsigned int iz = pixtreme_wrapped_floor(floor_z);
    const float fx = (float)(finite_x - floor_x);
    const float fy = (float)(finite_y - floor_y);
    const float fz = (float)(z - floor_z);
    float values[8];

    #pragma unroll
    for (int dz = 0; dz < 2; ++dz) {
        #pragma unroll
        for (int dy = 0; dy < 2; ++dy) {
            #pragma unroll
            for (int dx = 0; dx < 2; ++dx) {
                values[(dz * 2 + dy) * 2 + dx] = pixtreme_gradient_dot(
                    ix + (unsigned int)dx,
                    iy + (unsigned int)dy,
                    iz + (unsigned int)dz,
                    stream,
                    fx - (float)dx,
                    fy - (float)dy,
                    fz - (float)dz
                );
            }
        }
    }

    const float u = pixtreme_fade(fx);
    const float v = pixtreme_fade(fy);
    const float t = pixtreme_fade(fz);
    const float x00 = pixtreme_lerp(values[0], values[1], u);
    const float x10 = pixtreme_lerp(values[2], values[3], u);
    const float x01 = pixtreme_lerp(values[4], values[5], u);
    const float x11 = pixtreme_lerp(values[6], values[7], u);
    const float y0 = pixtreme_lerp(x00, x10, v);
    const float y1 = pixtreme_lerp(x01, x11, v);
    return pixtreme_lerp(y0, y1, t);
}

__device__ float pixtreme_gaussian_lattice(
    const unsigned int x,
    const unsigned int y,
    const unsigned int z,
    const unsigned int stream
) {
    const uint4 hashed = pixtreme_pcg4d(make_uint4(x, y, z, stream));
    const double u1 = ((double)hashed.x + 0.5) * (1.0 / 4294967296.0);
    const double u2 = ((double)hashed.y + 0.5) * (1.0 / 4294967296.0);
    return (float)(
        sqrt(-2.0 * log(u1)) *
        cos(6.283185307179586476925286766559 * u2)
    );
}

__device__ float pixtreme_grain_noise(
    const double x,
    const double y,
    const double z,
    const unsigned int stream
) {
    const double floor_x = floor(x);
    const double floor_y = floor(y);
    const double floor_z = floor(z);
    const unsigned int ix = pixtreme_wrapped_floor(floor_x);
    const unsigned int iy = pixtreme_wrapped_floor(floor_y);
    const unsigned int iz = pixtreme_wrapped_floor(floor_z);
    const float fx = (float)(x - floor_x);
    const float fy = (float)(y - floor_y);
    const float fz = (float)(z - floor_z);
    float values[8];

    #pragma unroll
    for (int dz = 0; dz < 2; ++dz) {
        #pragma unroll
        for (int dy = 0; dy < 2; ++dy) {
            #pragma unroll
            for (int dx = 0; dx < 2; ++dx) {
                values[(dz * 2 + dy) * 2 + dx] = pixtreme_gaussian_lattice(
                    ix + (unsigned int)dx,
                    iy + (unsigned int)dy,
                    iz + (unsigned int)dz,
                    stream
                );
            }
        }
    }

    const float x00 = pixtreme_lerp(values[0], values[1], fx);
    const float x10 = pixtreme_lerp(values[2], values[3], fx);
    const float x01 = pixtreme_lerp(values[4], values[5], fx);
    const float x11 = pixtreme_lerp(values[6], values[7], fx);
    const float y0 = pixtreme_lerp(x00, x10, fy);
    const float y1 = pixtreme_lerp(x01, x11, fy);
    return pixtreme_lerp(y0, y1, fz);
}

__device__ float pixtreme_clamp01(const float value) {
    return value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
}

extern "C" __global__ void pixtreme_noise_hash(
    const unsigned int* __restrict__ input,
    unsigned int* __restrict__ output,
    const long long count
) {
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const long long offset = index * 4LL;
    const uint4 hashed = pixtreme_pcg4d(make_uint4(
        input[offset],
        input[offset + 1],
        input[offset + 2],
        input[offset + 3]
    ));
    output[offset] = hashed.x;
    output[offset + 1] = hashed.y;
    output[offset + 2] = hashed.z;
    output[offset + 3] = hashed.w;
}

extern "C" __global__ void pixtreme_generate_gradient_noise(
    float* __restrict__ output,
    const long long width,
    const long long height,
    const double scale,
    const long long octaves,
    const double lacunarity,
    const double gain,
    const unsigned int seed,
    const double evolution,
    const int turbulent
) {
    const long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long y = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    double frequency = 1.0;
    double previous_weight_over_current = 0.0;
    float average = 0.0f;
    for (long long octave = 0; octave < octaves; ++octave) {
        float sample = pixtreme_gradient_noise(
            (((double)x + 0.5) / scale) * frequency,
            (((double)y + 0.5) / scale) * frequency,
            evolution,
            pixtreme_stream(seed, (unsigned long long)octave, 0u)
        );
        if (turbulent != 0) {
            sample = fabsf(sample);
        }
        const double contribution = 1.0 / (previous_weight_over_current + 1.0);
        average += (float)contribution * (sample - average);
        if (gain == 0.0) {
            break;
        }
        previous_weight_over_current = (previous_weight_over_current + 1.0) / gain;
        frequency *= lacunarity;
    }

    const float normalized = average * (2.0f / 1.7320508075688772935f);
    output[y * width + x] = pixtreme_clamp01(
        turbulent != 0 ? normalized : 0.5f + 0.5f * normalized
    );
}

extern "C" __global__ void pixtreme_generate_grain(
    float* __restrict__ output,
    const long long width,
    const long long height,
    const int channel_count,
    const double intensity,
    const double size,
    const unsigned int seed,
    const double evolution
) {
    const long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long y = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const double noise_x = ((double)x + 0.5) / size - 0.5;
    const double noise_y = ((double)y + 0.5) / size - 0.5;
    const long long offset = (y * width + x) * (long long)channel_count;
    for (int channel = 0; channel < channel_count; ++channel) {
        const float gaussian = pixtreme_grain_noise(
            noise_x,
            noise_y,
            evolution,
            pixtreme_stream(seed, 0u, (unsigned int)channel)
        );
        output[offset + channel] = pixtreme_clamp01(
            0.5f + (float)(intensity * 0.5 / 3.0) * gaussian
        );
    }
}
"""


@lru_cache(maxsize=1)
def _gradient_noise_kernel() -> cp.RawKernel:
    return cp.RawKernel(_NOISE_KERNEL_SOURCE, "pixtreme_generate_gradient_noise")


@lru_cache(maxsize=1)
def _grain_kernel() -> cp.RawKernel:
    return cp.RawKernel(_NOISE_KERNEL_SOURCE, "pixtreme_generate_grain")


def _octaves(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            _actionable_error(
                why="octaves must be an integer greater than or equal to 1",
                what=f"received octaves={value!r}",
                how="pass octaves as a non-bool int of at least 1",
            )
        )
    return value


def _seed(value: object) -> int:
    if value is None:
        return int(np.random.default_rng().integers(0, _UINT32_MODULUS, dtype=np.uint32))
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            _actionable_error(
                why="seed must be an int or None",
                what=f"received seed={value!r}",
                how="pass an integer for a deterministic realization or None for local entropy",
            )
        )
    return value & (_UINT32_MODULUS - 1)


def _grid(width: int, height: int) -> tuple[int, int]:
    return (
        (width + _NOISE_BLOCK[0] - 1) // _NOISE_BLOCK[0],
        (height + _NOISE_BLOCK[1] - 1) // _NOISE_BLOCK[1],
    )


def _generate_gradient_noise(
    *,
    width: int,
    height: int,
    scale: float,
    octaves: int,
    lacunarity: float,
    gain: float,
    seed: int,
    evolution: float,
    colorspace: str,
    gamma: str,
    turbulent: bool,
) -> Frame:
    output = cp.empty((height, width, 1), dtype=cp.float32)
    _gradient_noise_kernel()(
        _grid(width, height),
        _NOISE_BLOCK,
        (
            output,
            np.int64(width),
            np.int64(height),
            np.float64(scale),
            np.int64(octaves),
            np.float64(lacunarity),
            np.float64(gain),
            np.uint32(seed),
            np.float64(evolution),
            np.int32(turbulent),
        ),
    )
    return Frame(data=output, colorspace=colorspace, gamma=gamma, channels=("Y",))


def fractal_noise(
    *,
    width: int,
    height: int,
    scale: float,
    octaves: int = 4,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    seed: int | None = 0,
    evolution: float = 0.0,
    colorspace: str,
    gamma: str = "linear",
) -> Frame:
    """Generate normalized fractal gradient noise on the GPU.

    Pixel ``(i, j)`` uses centers ``i + 0.5`` and ``j + 0.5`` divided by
    ``scale``. ``seed`` selects a deterministic realization; ``None`` selects
    one from local entropy. ``evolution`` is a continuous third-coordinate
    phase shared by every octave. Octave frequencies use ``lacunarity`` and
    amplitudes use ``gain``. The weighted result is centered at 0.5 and
    normalized to ``[0, 1]`` with ``C = sqrt(3) / 2``.

    Each octave uses the ordinary xy coordinate while both axes remain finite
    and below ``2**53``, the largest double value with unit lattice precision.
    If frequency growth or division by ``scale`` makes an axis non-finite or
    reaches that limit, that axis evaluates at lattice origin ``0.0`` for the
    octave. The seed-derived octave stream and finite ``evolution`` phase stay
    active, giving a deterministic finite limit that may be constant along one
    or both image axes.

    The returned fp32 Frame reports the requested colorspace and gamma,
    declares channels ``("Y",)``, and owns a new C-contiguous allocation.
    """
    checked_width = _dimension(width, name="width")
    checked_height = _dimension(height, name="height")
    checked_scale = _positive_real(scale, name="scale")
    checked_octaves = _octaves(octaves)
    checked_lacunarity = _positive_real(lacunarity, name="lacunarity")
    checked_gain = _bounded_real(
        gain,
        name="gain",
        minimum=0.0,
        why="gain must be greater than or equal to 0",
        how="pass a finite non-negative real number for gain",
    )
    checked_seed = _seed(seed)
    checked_evolution = _finite_real(evolution, name="evolution")
    checked_colorspace, checked_gamma = _metadata(colorspace, gamma)
    return _generate_gradient_noise(
        width=checked_width,
        height=checked_height,
        scale=checked_scale,
        octaves=checked_octaves,
        lacunarity=checked_lacunarity,
        gain=checked_gain,
        seed=checked_seed,
        evolution=checked_evolution,
        colorspace=checked_colorspace,
        gamma=checked_gamma,
        turbulent=False,
    )


def turbulent_noise(
    *,
    width: int,
    height: int,
    scale: float,
    octaves: int = 4,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    seed: int | None = 0,
    evolution: float = 0.0,
    colorspace: str,
    gamma: str = "linear",
) -> Frame:
    """Generate normalized absolute-value turbulent gradient noise on the GPU.

    Pixel ``(i, j)`` uses centers ``i + 0.5`` and ``j + 0.5`` divided by
    ``scale``. ``seed`` selects a deterministic realization; ``None`` selects
    one from local entropy. ``evolution`` is a continuous third-coordinate
    phase shared by every octave. Absolute octave values are weighted by
    ``gain``, with frequencies set by ``lacunarity``, then normalized to
    ``[0, 1]`` with ``C = sqrt(3) / 2``.

    Each octave uses the ordinary xy coordinate while both axes remain finite
    and below ``2**53``, the largest double value with unit lattice precision.
    If frequency growth or division by ``scale`` makes an axis non-finite or
    reaches that limit, that axis evaluates at lattice origin ``0.0`` for the
    octave. The seed-derived octave stream and finite ``evolution`` phase stay
    active, giving a deterministic finite limit that may be constant along one
    or both image axes.

    The returned fp32 Frame reports the requested colorspace and gamma,
    declares channels ``("Y",)``, and owns a new C-contiguous allocation.
    """
    checked_width = _dimension(width, name="width")
    checked_height = _dimension(height, name="height")
    checked_scale = _positive_real(scale, name="scale")
    checked_octaves = _octaves(octaves)
    checked_lacunarity = _positive_real(lacunarity, name="lacunarity")
    checked_gain = _bounded_real(
        gain,
        name="gain",
        minimum=0.0,
        why="gain must be greater than or equal to 0",
        how="pass a finite non-negative real number for gain",
    )
    checked_seed = _seed(seed)
    checked_evolution = _finite_real(evolution, name="evolution")
    checked_colorspace, checked_gamma = _metadata(colorspace, gamma)
    return _generate_gradient_noise(
        width=checked_width,
        height=checked_height,
        scale=checked_scale,
        octaves=checked_octaves,
        lacunarity=checked_lacunarity,
        gain=checked_gain,
        seed=checked_seed,
        evolution=checked_evolution,
        colorspace=checked_colorspace,
        gamma=checked_gamma,
        turbulent=True,
    )


def grain(
    *,
    width: int,
    height: int,
    intensity: float = 0.1,
    size: float = 1.0,
    monochromatic: bool = True,
    seed: int | None = 0,
    evolution: float = 0.0,
    colorspace: str,
    gamma: str = "linear",
) -> Frame:
    """Generate Gaussian grain with continuous spatial and evolution interpolation.

    Pixel centers ``i + 0.5`` and ``j + 0.5`` sample a lattice whose xy spacing
    is ``size``; size 1 places one lattice realization at each pixel center.
    ``seed`` selects the realization and ``None`` uses local entropy.
    ``evolution`` is the continuous third-coordinate phase, with adjacent
    integer phases selecting independent lattice layers. Color mode folds each
    RGB channel into an independent stream.

    Box-Muller Gaussian values use a three standard deviation normalization:
    ``0.5 + intensity * 0.5 * g / 3``. Output is clipped to ``[0, 1]``; at
    intensity 1 the statistical clip residual is approximately 0.27%.
    The returned fp32 Frame reports the requested colorspace and gamma,
    declares Y or RGB channels, and owns a new C-contiguous allocation.
    """
    checked_width = _dimension(width, name="width")
    checked_height = _dimension(height, name="height")
    checked_intensity = _bounded_real(
        intensity,
        name="intensity",
        minimum=0.0,
        why="intensity must be greater than or equal to 0",
        how="pass a finite non-negative real number for intensity",
    )
    checked_size = _positive_real(size, name="size")
    checked_monochromatic = _strict_bool(
        monochromatic,
        name="monochromatic",
        why="monochromatic must be a bool",
        how="pass monochromatic=True or monochromatic=False",
    )
    checked_seed = _seed(seed)
    checked_evolution = _finite_real(evolution, name="evolution")
    checked_colorspace, checked_gamma = _metadata(colorspace, gamma)
    channel_count = 1 if checked_monochromatic else 3
    output = cp.empty((checked_height, checked_width, channel_count), dtype=cp.float32)
    _grain_kernel()(
        _grid(checked_width, checked_height),
        _NOISE_BLOCK,
        (
            output,
            np.int64(checked_width),
            np.int64(checked_height),
            np.int32(channel_count),
            np.float64(checked_intensity),
            np.float64(checked_size),
            np.uint32(checked_seed),
            np.float64(checked_evolution),
        ),
    )
    return Frame(
        data=output,
        colorspace=checked_colorspace,
        gamma=checked_gamma,
        channels=("Y",) if checked_monochromatic else ("R", "G", "B"),
    )
