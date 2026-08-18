"""Specification, contract, and independent-oracle tests for procedural noise generators."""

from __future__ import annotations

import inspect
import math
from typing import Any

import noise_test_harness as noise_harness
import numpy as np
import pytest

import pixtreme as px

NOISE_NAMES = (
    "fractal_noise",
    "turbulent_noise",
    "grain",
)
_UINT32_MASK = (1 << 32) - 1
_UINT32_SCALE = float(1 << 32)
_GRADIENT_DENOMINATOR = float(_UINT32_MASK)
_NORMALIZATION_C = math.sqrt(3.0) / 2.0
# The NumPy oracle evaluates the published equations in float64, while CUDA
# evaluates interpolation and transcendental steps in fp32. The allowance is
# bounded to the accumulated fp32 operation-order and libdevice differences.
_INTERPOLATED_ATOL = 3e-5


def _host(frame: px.core.Frame) -> np.ndarray:
    import cupy as cp

    return cp.asnumpy(frame.data)


def _assert_actionable(error: pytest.ExceptionInfo[ValueError]) -> None:
    message = str(error.value)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message


def _base_kwargs(name: str) -> dict[str, Any]:
    common = {
        "width": 12,
        "height": 9,
        "seed": 17,
        "evolution": 0.375,
        "colorspace": "ACEScg",
    }
    if name == "grain":
        return common | {"intensity": 0.6, "size": 1.75}
    return common | {"scale": 5.5, "octaves": 3, "lacunarity": 1.75, "gain": 0.4}


def _u32(value: int) -> int:
    return value & _UINT32_MASK


def _pcg4d(value: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Independent scalar transcription of Jarzynski/Olano PCG4D."""
    x, y, z, w = (_u32(component * 1_664_525 + 1_013_904_223) for component in value)
    x = _u32(x + y * w)
    y = _u32(y + z * x)
    z = _u32(z + x * y)
    w = _u32(w + y * z)
    x ^= x >> 16
    y ^= y >> 16
    z ^= z >> 16
    w ^= w >> 16
    x = _u32(x + y * w)
    y = _u32(y + z * x)
    z = _u32(z + x * y)
    w = _u32(w + y * z)
    return x, y, z, w


def _stream(seed: int, *, octave: int = 0, channel: int = 0) -> int:
    return _u32(seed) ^ _u32(0x9E3779B9 * octave) ^ _u32(0x85EBCA6B * channel)


def _gradient(ix: int, iy: int, iz: int, stream: int) -> np.ndarray:
    hashed = _pcg4d((_u32(ix), _u32(iy), _u32(iz), stream))
    vector = np.asarray(
        [2.0 * (hashed[index] / _GRADIENT_DENOMINATOR) - 1.0 for index in range(3)],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    return vector / norm


def _fade(value: float) -> float:
    return value**3 * (value * (value * 6.0 - 15.0) + 10.0)


def _gradient_noise(x: float, y: float, z: float, stream: int) -> float:
    ix = math.floor(x)
    iy = math.floor(y)
    iz = math.floor(z)
    fx = x - ix
    fy = y - iy
    fz = z - iz
    values = np.empty((2, 2, 2), dtype=np.float64)
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                offset = np.asarray((fx - dx, fy - dy, fz - dz), dtype=np.float64)
                values[dz, dy, dx] = float(np.dot(_gradient(ix + dx, iy + dy, iz + dz, stream), offset))
    u = _fade(fx)
    v = _fade(fy)
    t = _fade(fz)
    x00 = values[0, 0, 0] + u * (values[0, 0, 1] - values[0, 0, 0])
    x10 = values[0, 1, 0] + u * (values[0, 1, 1] - values[0, 1, 0])
    x01 = values[1, 0, 0] + u * (values[1, 0, 1] - values[1, 0, 0])
    x11 = values[1, 1, 0] + u * (values[1, 1, 1] - values[1, 1, 0])
    y0 = x00 + v * (x10 - x00)
    y1 = x01 + v * (x11 - x01)
    return y0 + t * (y1 - y0)


def _fractal_reference(
    *,
    width: int,
    height: int,
    scale: float,
    octaves: int,
    lacunarity: float,
    gain: float,
    seed: int,
    evolution: float,
    turbulent: bool,
) -> np.ndarray:
    output = np.empty((height, width, 1), dtype=np.float32)
    weights = np.asarray([gain**octave for octave in range(octaves)], dtype=np.float64)
    for j in range(height):
        for i in range(width):
            values = []
            for octave in range(octaves):
                frequency = lacunarity**octave
                value = _gradient_noise(
                    (i + 0.5) / scale * frequency,
                    (j + 0.5) / scale * frequency,
                    evolution,
                    _stream(seed, octave=octave),
                )
                values.append(abs(value) if turbulent else value)
            combined = float(np.dot(weights, np.asarray(values, dtype=np.float64)) / weights.sum())
            normalized = combined / _NORMALIZATION_C
            output[j, i, 0] = normalized if turbulent else 0.5 + 0.5 * normalized
    return output


def _gaussian_lattice_value(ix: int, iy: int, iz: int, stream: int) -> float:
    hashed = _pcg4d((_u32(ix), _u32(iy), _u32(iz), stream))
    u1 = (hashed[0] + 0.5) / _UINT32_SCALE
    u2 = (hashed[1] + 0.5) / _UINT32_SCALE
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _grain_sample(x: float, y: float, z: float, stream: int) -> float:
    ix = math.floor(x)
    iy = math.floor(y)
    iz = math.floor(z)
    fx = x - ix
    fy = y - iy
    fz = z - iz
    values = np.empty((2, 2, 2), dtype=np.float64)
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                values[dz, dy, dx] = _gaussian_lattice_value(ix + dx, iy + dy, iz + dz, stream)
    x00 = values[0, 0, 0] + fx * (values[0, 0, 1] - values[0, 0, 0])
    x10 = values[0, 1, 0] + fx * (values[0, 1, 1] - values[0, 1, 0])
    x01 = values[1, 0, 0] + fx * (values[1, 0, 1] - values[1, 0, 0])
    x11 = values[1, 1, 0] + fx * (values[1, 1, 1] - values[1, 1, 0])
    y0 = x00 + fy * (x10 - x00)
    y1 = x01 + fy * (x11 - x01)
    return y0 + fz * (y1 - y0)


def _grain_reference(
    *,
    width: int,
    height: int,
    intensity: float,
    size: float,
    monochromatic: bool,
    seed: int,
    evolution: float,
) -> np.ndarray:
    channels = 1 if monochromatic else 3
    output = np.empty((height, width, channels), dtype=np.float32)
    for j in range(height):
        for i in range(width):
            # A half-pixel lattice phase makes size=1 place one independent
            # Gaussian lattice value at every pixel center, as required by the
            # per-pixel and statistical contracts.
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            for channel in range(channels):
                gaussian = _grain_sample(x, y, evolution, _stream(seed, channel=channel))
                output[j, i, channel] = np.clip(0.5 + intensity * 0.5 * gaussian / 3.0, 0.0, 1.0)
    return output


def test_noise_public_signatures_are_keyword_only_and_minimal() -> None:
    """v1-derivative-filters acceptance 17: noise stays in the expanded 68-point public surface."""
    expected = {
        "fractal_noise": (
            ("width", inspect.Parameter.empty),
            ("height", inspect.Parameter.empty),
            ("scale", inspect.Parameter.empty),
            ("octaves", 4),
            ("lacunarity", 2.0),
            ("gain", 0.5),
            ("seed", 0),
            ("evolution", 0.0),
            ("colorspace", inspect.Parameter.empty),
            ("gamma", "linear"),
        ),
        "turbulent_noise": (
            ("width", inspect.Parameter.empty),
            ("height", inspect.Parameter.empty),
            ("scale", inspect.Parameter.empty),
            ("octaves", 4),
            ("lacunarity", 2.0),
            ("gain", 0.5),
            ("seed", 0),
            ("evolution", 0.0),
            ("colorspace", inspect.Parameter.empty),
            ("gamma", "linear"),
        ),
        "grain": (
            ("width", inspect.Parameter.empty),
            ("height", inspect.Parameter.empty),
            ("intensity", 0.1),
            ("size", 1.0),
            ("monochromatic", True),
            ("seed", 0),
            ("evolution", 0.0),
            ("colorspace", inspect.Parameter.empty),
            ("gamma", "linear"),
        ),
    }
    for name, parameters in expected.items():
        function = getattr(px.generate, name)
        signature = inspect.signature(function)
        assert tuple((parameter.name, parameter.default) for parameter in signature.parameters.values()) == parameters
        assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in signature.parameters.values())
        assert name in px.generate.__all__
    assert len(px.generate.__all__) == 7


@pytest.mark.parametrize("name", NOISE_NAMES)
@pytest.mark.parametrize("axis,value", (("width", 0), ("height", -1), ("width", 1.5), ("height", True)))
def test_noise_dimensions_are_positive_non_bool_integers(name: str, axis: str, value: object) -> None:
    """v1-noise acceptance 2: dimensions reject non-positive and non-integer values with recovery guidance."""
    with pytest.raises(ValueError) as error:
        getattr(px.generate, name)(**(_base_kwargs(name) | {axis: value}))
    _assert_actionable(error)


@pytest.mark.parametrize(
    ("name", "axis", "value"),
    (
        ("fractal_noise", "scale", 0.0),
        ("turbulent_noise", "scale", math.inf),
        ("fractal_noise", "octaves", 0),
        ("turbulent_noise", "octaves", True),
        ("fractal_noise", "lacunarity", -1.0),
        ("turbulent_noise", "lacunarity", math.nan),
        ("fractal_noise", "gain", -0.1),
        ("turbulent_noise", "gain", math.inf),
        ("grain", "intensity", -0.1),
        ("grain", "intensity", math.nan),
        ("grain", "size", 0.0),
        ("grain", "size", math.inf),
    ),
)
def test_noise_shape_and_amplitude_parameters_fail_fast(name: str, axis: str, value: object) -> None:
    """v1-noise acceptance 3-6 and 9-10: shape and amplitude inputs enforce their finite numeric domains."""
    with pytest.raises(ValueError) as error:
        getattr(px.generate, name)(**(_base_kwargs(name) | {axis: value}))
    _assert_actionable(error)


@pytest.mark.parametrize("name", NOISE_NAMES)
@pytest.mark.parametrize("value", (True, 1.5, "7"))
def test_noise_seed_rejects_bool_and_non_integer_types(name: str, value: object) -> None:
    """v1-noise acceptance 7: seed accepts only int or None and reports an actionable error."""
    with pytest.raises(ValueError) as error:
        getattr(px.generate, name)(**(_base_kwargs(name) | {"seed": value}))
    _assert_actionable(error)


@pytest.mark.parametrize("name", NOISE_NAMES)
@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf, True))
def test_noise_evolution_requires_a_finite_non_bool_real(name: str, value: object) -> None:
    """v1-noise acceptance 8: evolution accepts finite positive or negative phases and rejects other values."""
    with pytest.raises(ValueError) as error:
        getattr(px.generate, name)(**(_base_kwargs(name) | {"evolution": value}))
    _assert_actionable(error)


@pytest.mark.parametrize("value", (0, 1, "true", None))
def test_grain_monochromatic_accepts_bool_only(value: object) -> None:
    """v1-noise acceptance 11: monochromatic is a strict bool axis."""
    with pytest.raises(ValueError) as error:
        px.generate.grain(**(_base_kwargs("grain") | {"monochromatic": value}))
    _assert_actionable(error)


@pytest.mark.parametrize("name", NOISE_NAMES)
@pytest.mark.parametrize(("axis", "token"), (("colorspace", "acescg"), ("gamma", "Linear")))
def test_noise_metadata_tokens_fail_fast(name: str, axis: str, token: str) -> None:
    """v1-noise acceptance 12: colorspace and gamma share the closed case-sensitive vocabulary."""
    with pytest.raises(ValueError) as error:
        getattr(px.generate, name)(**(_base_kwargs(name) | {axis: token}))
    _assert_actionable(error)


@pytest.mark.parametrize("name", NOISE_NAMES)
def test_noise_outputs_are_new_contiguous_fp32_frames_with_requested_metadata(name: str) -> None:
    """v1-noise acceptance 13-16: output ownership, layout, dtype, channels, dimensions, and metadata are fixed."""
    kwargs = _base_kwargs(name) | {"colorspace": "S-Gamut3", "gamma": "s-log3"}
    first = getattr(px.generate, name)(**kwargs)
    second = getattr(px.generate, name)(**kwargs)
    expected_channels = ("Y",)
    assert first.shape == (9, 12, 1)
    assert first.dtype == np.dtype(np.float32)
    assert first.data.flags.c_contiguous
    assert first.data.data.ptr != second.data.data.ptr
    assert (first.colorspace, first.gamma, first.channels) == ("S-Gamut3", "s-log3", expected_channels)


def test_color_grain_has_three_independent_rgb_channels() -> None:
    """v1-noise acceptance 15 and 26: color grain declares RGB and folds channel identity into independent streams."""
    result = px.generate.grain(**(_base_kwargs("grain") | {"monochromatic": False, "width": 64, "height": 64}))
    host = _host(result)
    assert result.shape == (64, 64, 3)
    assert result.channels == ("R", "G", "B")
    assert not np.array_equal(host[..., 0], host[..., 1])
    assert not np.array_equal(host[..., 1], host[..., 2])


@pytest.mark.parametrize("name", NOISE_NAMES)
def test_fixed_seed_calls_are_bit_deterministic(name: str) -> None:
    """v1-noise acceptance 17: identical fixed-seed calls return bit-identical fp32 values."""
    kwargs = _base_kwargs(name)
    assert np.array_equal(_host(getattr(px.generate, name)(**kwargs)), _host(getattr(px.generate, name)(**kwargs)))


@pytest.mark.parametrize(
    ("scale", "lacunarity", "octaves"),
    (
        (1e-308, 2.0, 4),
        (5.5, 1e300, 4),
        (1e-308, 1e300, 4),
        (5.5, 10.0, 64),
        (math.ulp(0.0), math.ulp(0.0), 4),
    ),
)
def test_fractal_noise_extreme_positive_finite_inputs_remain_finite_and_deterministic(
    scale: float,
    lacunarity: float,
    octaves: int,
) -> None:
    """REQ-TEST-001; issue #8 acceptance 1 and 3: extreme accepted inputs stay finite and deterministic."""
    kwargs = {
        "width": 2,
        "height": 2,
        "scale": scale,
        "octaves": octaves,
        "lacunarity": lacunarity,
        "gain": 0.5,
        "seed": 17,
        "evolution": 0.375,
        "colorspace": "ACEScg",
    }
    first = _host(px.generate.fractal_noise(**kwargs))
    second = _host(px.generate.fractal_noise(**kwargs))
    assert np.all(np.isfinite(first))
    assert np.array_equal(first, second)


def test_fractal_noise_extreme_xy_uses_documented_lattice_origin_limit() -> None:
    """REQ-TEST-003; issue #8 acceptance 1 and 4: overflowed xy evaluates the independent origin oracle."""
    seed = 17
    evolution = 0.375
    octaves = 4
    gain = 0.5
    actual = _host(
        px.generate.fractal_noise(
            width=3,
            height=2,
            scale=1e-308,
            octaves=octaves,
            lacunarity=2.0,
            gain=gain,
            seed=seed,
            evolution=evolution,
            colorspace="ACEScg",
        )
    )
    weights = np.asarray([gain**octave for octave in range(octaves)], dtype=np.float64)
    samples = np.asarray(
        [_gradient_noise(0.0, 0.0, evolution, _stream(seed, octave=octave)) for octave in range(octaves)],
        dtype=np.float64,
    )
    expected = np.float32(
        np.clip(0.5 + 0.5 * float(np.dot(weights, samples) / weights.sum()) / _NORMALIZATION_C, 0.0, 1.0)
    )
    assert np.array_equal(actual, np.full_like(actual, actual[0, 0, 0]))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=_INTERPOLATED_ATOL)


def test_fractal_noise_regular_domain_remains_bit_identical_characterization() -> None:
    """characterization: issue #8 acceptance 2 freezes regular-domain bits while overflow handling is repaired.

    The independent v1-noise oracle establishes correctness to its documented fp32 tolerance; this exact snapshot
    separately freezes the current CUDA operation order and is replaced only if that public normal-domain behavior changes.
    """
    actual = _host(
        px.generate.fractal_noise(
            width=3,
            height=2,
            scale=5.5,
            octaves=3,
            lacunarity=1.75,
            gain=0.4,
            seed=17,
            evolution=0.375,
            colorspace="ACEScg",
        )
    )
    expected_bits = np.asarray(
        (1057036128, 1055642634, 1055343856, 1057247706, 1055098936, 1053423683),
        dtype=np.uint32,
    ).reshape(2, 3, 1)
    assert np.array_equal(actual.view(np.uint32), expected_bits)


def test_turbulent_noise_regular_domain_remains_bit_identical_characterization() -> None:
    """characterization: the current turbulent-noise CUDA operation order stays bit-identical.

    The independent v1-noise oracle establishes correctness to its documented fp32 tolerance; this exact snapshot
    separately freezes the public regular-domain bits while the kernel evaluation strategy is optimized.
    """
    actual = _host(
        px.generate.turbulent_noise(
            width=3,
            height=2,
            scale=5.5,
            octaves=3,
            lacunarity=1.75,
            gain=0.4,
            seed=17,
            evolution=0.375,
            colorspace="ACEScg",
        )
    )
    expected_bits = np.asarray(
        (1035864337, 1037207217, 1042751870, 1033789575, 1038335555, 1045962483),
        dtype=np.uint32,
    ).reshape(2, 3, 1)
    assert np.array_equal(actual.view(np.uint32), expected_bits)


@pytest.mark.parametrize(
    ("name", "expected_values"),
    (
        (
            "fractal_noise",
            (1057158808, 1057127417, 1057058467, 1057242783, 1057208144, 1057129380),
        ),
        (
            "turbulent_noise",
            (1042839283, 1042564057, 1042030801, 1042620985, 1042317180, 1041697454),
        ),
    ),
)
def test_tiled_gradient_noise_output_remains_bit_identical_characterization(
    name: str,
    expected_values: tuple[int, ...],
) -> None:
    """characterization: the tiled scale-64 CUDA path stays bit-identical to the public operation order.

    The independent v1-noise oracle establishes correctness to its documented fp32 tolerance; this exact snapshot
    separately freezes the shared-lattice path for both signed and absolute-value accumulation.
    """
    actual = _host(
        getattr(px.generate, name)(
            width=3,
            height=2,
            scale=64.0,
            octaves=4,
            lacunarity=2.0,
            gain=0.5,
            seed=17,
            evolution=0.375,
            colorspace="ACEScg",
        )
    )
    expected_bits = np.asarray(expected_values, dtype=np.uint32).reshape(2, 3, 1)
    assert np.array_equal(actual.view(np.uint32), expected_bits)


def test_color_grain_lattice_aligned_output_remains_bit_identical_characterization() -> None:
    """characterization: the current RGB grain bits stay fixed at the size-one lattice-aligned fast-path domain.

    The independent v1-noise oracle establishes correctness to its documented fp32 tolerance; this exact snapshot
    separately freezes every channel's current Box-Muller and interpolation operation order during optimization.
    """
    actual = _host(
        px.generate.grain(
            width=3,
            height=2,
            intensity=0.37,
            size=1.0,
            monochromatic=False,
            seed=17,
            evolution=0.375,
            colorspace="ACEScg",
        )
    )
    expected_bits = np.asarray(
        (
            1056549594,
            1055809679,
            1057291404,
            1056167636,
            1057133042,
            1056965549,
            1056146083,
            1058016404,
            1057361045,
            1057755079,
            1053431044,
            1055930865,
            1057335909,
            1057507141,
            1058367003,
            1058204314,
            1057979738,
            1058215725,
        ),
        dtype=np.uint32,
    ).reshape(2, 3, 3)
    assert np.array_equal(actual.view(np.uint32), expected_bits)


def test_none_seed_uses_local_entropy_without_process_global_rng_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """v1-noise acceptance 18 and REQ-TEST-004: entropy realization is local and two calls differ."""
    import cupy as cp

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("process-global RNG mutation is forbidden")

    monkeypatch.setattr(np.random, "seed", forbidden)
    monkeypatch.setattr(cp.random, "seed", forbidden)
    kwargs = _base_kwargs("grain") | {"seed": None, "width": 64, "height": 64}
    first = _host(px.generate.grain(**kwargs))
    second = _host(px.generate.grain(**kwargs))
    assert not np.array_equal(first, second)


@pytest.mark.parametrize("name", NOISE_NAMES)
def test_noise_values_stay_in_unit_interval(name: str) -> None:
    """v1-noise acceptance 19 and 29: every generator stays in the documented normalized interval."""
    kwargs = _base_kwargs(name) | {"width": 96, "height": 72}
    if name == "grain":
        kwargs["intensity"] = 3.0
    result = _host(getattr(px.generate, name)(**kwargs))
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_pcg4d_integer_hash_matches_the_independent_reference_exactly() -> None:
    """v1-noise acceptance 21 and 28: device PCG4D uint32 output exactly matches the paper-derived oracle."""
    import cupy as cp

    vectors = np.asarray(
        (
            (0, 0, 0, 0),
            (1, 2, 3, 4),
            (_UINT32_MASK, 0x80000000, 0x12345678, 0x9ABCDEF0),
            (17, 23, 42, _stream(-9, octave=3, channel=2)),
        ),
        dtype=np.uint32,
    )
    device_input = cp.asarray(vectors)
    device_output = cp.empty_like(device_input)
    noise_harness._hash_kernel()(
        (1,),
        (vectors.shape[0],),
        (device_input, device_output, np.int64(vectors.shape[0])),
    )
    expected = np.asarray([_pcg4d(tuple(int(component) for component in row)) for row in vectors], dtype=np.uint32)
    assert np.array_equal(cp.asnumpy(device_output), expected)


def test_gradient_noise_lattice_points_have_exact_normalized_values() -> None:
    """v1-noise acceptance 20, 22-24, and 28: integer lattice evaluation has exact zero gradient contribution."""
    common = {
        "width": 1,
        "height": 1,
        "scale": 0.5,
        "octaves": 1,
        "seed": 5,
        "evolution": 0.0,
        "colorspace": "ACEScg",
    }
    assert _host(px.generate.fractal_noise(**common)).item() == np.float32(0.5)
    assert _host(px.generate.turbulent_noise(**common)).item() == np.float32(0.0)


@pytest.mark.parametrize(("name", "turbulent"), (("fractal_noise", False), ("turbulent_noise", True)))
def test_gradient_noise_matches_independent_numpy_equations(name: str, turbulent: bool) -> None:
    """v1-noise acceptance 20-24 and 28: representative multi-octave output matches the independent NumPy oracle."""
    kwargs = {
        "width": 7,
        "height": 5,
        "scale": 3.25,
        "octaves": 4,
        "lacunarity": 1.8,
        "gain": 0.55,
        "seed": -1234567890123,
        "evolution": -0.375,
        "colorspace": "Rec.2020",
    }
    actual = _host(getattr(px.generate, name)(**kwargs))
    expected = _fractal_reference(
        width=kwargs["width"],
        height=kwargs["height"],
        scale=kwargs["scale"],
        octaves=kwargs["octaves"],
        lacunarity=kwargs["lacunarity"],
        gain=kwargs["gain"],
        seed=kwargs["seed"],
        evolution=kwargs["evolution"],
        turbulent=turbulent,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=_INTERPOLATED_ATOL)


@pytest.mark.parametrize("monochromatic", (True, False))
def test_grain_matches_independent_numpy_equations(monochromatic: bool) -> None:
    """v1-noise acceptance 20-21 and 25-28: grain interpolation and channel streams match the independent oracle."""
    kwargs = {
        "width": 6,
        "height": 4,
        "intensity": 0.8,
        "size": 2.25,
        "monochromatic": monochromatic,
        "seed": 0x1_2345_6789,
        "evolution": 0.625,
        "colorspace": "sRGB",
    }
    actual = _host(px.generate.grain(**kwargs))
    expected = _grain_reference(
        width=kwargs["width"],
        height=kwargs["height"],
        intensity=kwargs["intensity"],
        size=kwargs["size"],
        monochromatic=kwargs["monochromatic"],
        seed=kwargs["seed"],
        evolution=kwargs["evolution"],
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=_INTERPOLATED_ATOL)


@pytest.mark.parametrize("name", NOISE_NAMES)
def test_evolution_changes_continuously_for_small_phase_steps(name: str) -> None:
    """v1-noise acceptance 22, 25, and 29: a small phase change produces a bounded continuous output change."""
    kwargs = _base_kwargs(name) | {"width": 64, "height": 48, "evolution": -0.125}
    first = _host(getattr(px.generate, name)(**kwargs))
    second = _host(getattr(px.generate, name)(**(kwargs | {"evolution": -0.1249})))
    assert np.max(np.abs(second - first)) < 0.01
    assert not np.array_equal(first, second)


@pytest.mark.parametrize("name", ("fractal_noise", "turbulent_noise"))
def test_different_noise_seeds_are_effectively_uncorrelated(name: str) -> None:
    """v1-noise acceptance 21 and 29: separate seed realizations have no material linear correlation."""
    kwargs = _base_kwargs(name) | {"width": 192, "height": 128, "scale": 8.0}
    first = _host(getattr(px.generate, name)(**(kwargs | {"seed": 11}))).ravel()
    second = _host(getattr(px.generate, name)(**(kwargs | {"seed": 12}))).ravel()
    correlation = float(np.corrcoef(first, second)[0, 1])
    assert abs(correlation) < 0.15


def test_grain_statistics_match_the_three_sigma_normalization() -> None:
    """v1-noise acceptance 25 and 29: size-one grain has the specified mean, sigma, and clipping residual."""
    intensity = 1.0
    values = _host(
        px.generate.grain(
            width=512,
            height=512,
            intensity=intensity,
            size=1.0,
            seed=20260719,
            evolution=0.0,
            colorspace="ACEScg",
        )
    )[..., 0]
    clip_rate = float(np.mean((values == 0.0) | (values == 1.0)))
    assert float(values.mean()) == pytest.approx(0.5, abs=0.003)
    assert float(values.std()) == pytest.approx(intensity * 0.5 / 3.0, abs=0.003)
    assert clip_rate == pytest.approx(0.0027, abs=0.001)


def test_color_grain_channels_and_integer_evolution_steps_are_uncorrelated() -> None:
    """v1-noise acceptance 26-27 and 29: channel folds and adjacent integer phases select independent realizations."""
    kwargs = {
        "width": 256,
        "height": 256,
        "intensity": 1.0,
        "size": 1.0,
        "monochromatic": False,
        "seed": 91,
        "evolution": 2.0,
        "colorspace": "ACEScg",
    }
    first = _host(px.generate.grain(**kwargs))
    second = _host(px.generate.grain(**(kwargs | {"evolution": 3.0})))
    channel_correlation = np.corrcoef(first.reshape(-1, 3), rowvar=False)
    assert np.max(np.abs(channel_correlation - np.eye(3))) < 0.03
    assert abs(float(np.corrcoef(first[..., 0].ravel(), second[..., 0].ravel())[0, 1])) < 0.03


def test_noise_docstrings_are_self_contained_llm_readable_contracts() -> None:
    """v1-noise acceptance 31 / REQ-TEST-001; issue #8 acceptance 4: docstrings state the full numeric contract."""
    combined = "\n".join(inspect.getdoc(getattr(px.generate, name)) or "" for name in NOISE_NAMES).lower()
    for required in (
        "i + 0.5",
        "j + 0.5",
        "seed",
        "realization",
        "evolution",
        "continuous",
        "[0, 1]",
        "sqrt(3) / 2",
        "three",
        "standard deviation",
        "clip",
        "0.27%",
        "colorspace",
        "gamma",
        "channels",
        "new",
        "c-contiguous",
    ):
        assert required in combined
    fractal_contract = (inspect.getdoc(px.generate.fractal_noise) or "").lower()
    for required in ("non-finite", "lattice origin", "0.0", "axis", "octave", "evolution"):
        assert required in fractal_contract


def test_noise_generators_use_rawkernel_per_pixel_evaluation() -> None:
    """v1-noise acceptance 13, 20-27: structural contract fixes GPU RawKernel generation without host synthesis."""
    import pixtreme._generate.noise as noise_module

    source = inspect.getsource(noise_module)
    assert "cp.RawKernel" in source
    assert "cp.empty" in source
    assert "cp.asnumpy" not in source
    assert "cp.random" not in source


def test_noise_kernels_reuse_redundant_lattice_work() -> None:
    """REQ-TEST-003: structural contract requires tiled gradients and an aligned-grain lattice bypass."""
    import pixtreme._generate.noise as noise_module

    source = inspect.getsource(noise_module)
    assert "pixtreme_gradient_noise_tiled" in source
    assert "pixtreme_grain_noise_lattice_aligned" in source
    assert not noise_module._uses_tiled_gradient_kernel(scale=8.0, octaves=4, lacunarity=2.0, gain=0.5)
    assert noise_module._uses_tiled_gradient_kernel(scale=16.0, octaves=4, lacunarity=2.0, gain=0.5)
    assert not noise_module._uses_tiled_gradient_kernel(scale=16.0, octaves=4, lacunarity=4.0, gain=0.5)
    assert noise_module._uses_tiled_gradient_kernel(scale=16.0, octaves=1, lacunarity=4.0, gain=0.5)
    assert not noise_module._uses_tiled_gradient_kernel(scale=64.0, octaves=32, lacunarity=2.0, gain=0.5)
