"""Independent P210 code, sampling, and quality fixtures for P210 wire-format tests."""

from __future__ import annotations

import numpy as np
from test_from_format_spec import _axis_plan as upsample_axis
from test_to_format_spec import _downsample_reference as downsample

FROM_FILTERS = ("nearest", "bilinear", "bicubic", "b-spline", "mitchell", "lanczos2", "lanczos3", "lanczos4")
TO_FILTERS = ("nearest", "bilinear", "bicubic", "area")


def pack_planes(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    return np.concatenate((y.ravel(), np.stack((cb, cr), axis=-1).ravel())).astype(np.uint16)


def unpack_codes(words: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    codes = words.astype(np.uint16) >> 6
    y = codes[: height * width].reshape(height, width)
    uv = codes[height * width :].reshape(height, width // 2, 2)
    return y, uv[..., 0], uv[..., 1]


def asymmetric_words() -> np.ndarray:
    # Three rows distinguish 4:2:2 from 4:2:0; nonzero padding must be ignored.
    y = np.asarray(
        [[0, 1, 64, 940, 1023, 400], [940, 64, 512, 1022, 17, 1000], [3, 500, 901, 70, 800, 99]],
        dtype=np.uint16,
    )
    cb = np.asarray([[64, 512, 960], [0, 1023, 301], [171, 734, 992]], dtype=np.uint16)
    cr = np.asarray([[960, 512, 64], [1023, 0, 702], [850, 289, 31]], dtype=np.uint16)
    padding = np.arange(36, dtype=np.uint16) * 17 % 64
    return (pack_planes(y, cb, cr) << 6) | padding


def decode_values(codes: np.ndarray, range_token: str, component: int) -> np.ndarray:
    values = codes.astype(np.float64)
    if range_token == "full":
        return values / 1023.0
    return (values - 64.0) / (876.0 if component == 0 else 896.0)


def from_reference(words: np.ndarray, height: int, width: int, range_token: str, interpolation: str) -> np.ndarray:
    y, cb, cr = unpack_codes(words, height, width)
    result = np.empty((height, width, 3), dtype=np.float64)
    result[..., 0] = decode_values(y, range_token, 0)
    for component, codes in enumerate((cb, cr), start=1):
        plane = decode_values(codes, range_token, component)
        for row in range(height):
            for column in range(width):
                result[row, column, component] = sum(
                    plane[row, index] * weight
                    for index, weight in upsample_axis(column / 2.0, width // 2, interpolation)
                )
    return result


def to_reference(values: np.ndarray, range_token: str, interpolation: str) -> tuple[np.ndarray, np.ndarray]:
    planes = [values[..., 0].astype(np.float64)]
    planes.extend(
        downsample(values[..., channel], subsample_x=2, subsample_y=1, offset=(0.0, 0.0), interpolation=interpolation)
        for channel in (1, 2)
    )
    mapped = [
        plane * 1023.0 if range_token == "full" else plane * (876.0 if component == 0 else 896.0) + 64.0
        for component, plane in enumerate(planes)
    ]
    q64 = np.concatenate((mapped[0].ravel(), np.stack(mapped[1:], axis=-1).ravel()))
    rounded = np.copysign(np.floor(np.abs(q64) + 0.5), q64)
    return (np.clip(rounded, 0, 1023).astype(np.uint16) << 6), q64


def fixed_corpus() -> np.ndarray:
    values = np.random.default_rng(48).uniform(-0.125, 1.125, size=(17, 64, 3)).astype(np.float32)
    values[0, :6] = [(-0.125,) * 3, (0,) * 3, (0, 0.5, 1), (0.5,) * 3, (1,) * 3, (1.125,) * 3]
    return values


def assert_near_tie_codes(actual: np.ndarray, expected: np.ndarray, q64: np.ndarray, *, corpus: bool = False) -> None:
    """Accept only one-code fp32 differences at the AC-48-10 derived 1/512-code decision boundary."""
    assert actual.shape == expected.shape == q64.shape
    assert actual.dtype == np.uint16
    assert np.all((actual & 63) == 0), "P210 padding must be zero"
    differences = np.abs((actual >> 6).astype(np.int64) - (expected >> 6).astype(np.int64))
    assert differences.max() <= 1, f"maximum code difference: {differences.max()}"
    changed = differences != 0
    nearest_boundary = np.clip(np.floor(q64), 0, 1022) + 0.5
    assert np.all(np.abs(q64[changed] - nearest_boundary[changed]) <= 1 / 512), q64[changed]
    if corpus:
        assert actual.shape == (2176,)
        assert np.count_nonzero(changed) / actual.size <= 0.02


def quality_values(pattern: str) -> np.ndarray:
    y, x = np.mgrid[:17, :64]
    values = np.empty((17, 64, 3), dtype=np.float64)
    if pattern == "ramp":
        values[..., 0] = x / 63
        values[..., 1:] = 0.5
    elif pattern == "neutral":
        values[..., 0] = 0.25 + 0.5 * ((x + y) % 2)
        values[..., 1:] = 0.5
    elif pattern == "edge":
        values[..., 0] = 0.5
        values[..., 1] = np.where(x < 32, 0.25, 0.75)
        values[..., 2] = 1 - values[..., 1]
    elif pattern == "alternation":
        values[..., 0] = 0.5
        values[..., 1] = np.where(x % 2 == 0, 0.25, 0.75)
        values[..., 2] = 1 - values[..., 1]
    elif pattern == "zone":
        values[..., 0] = 0.5
        values[..., 1] = 0.5 + 0.25 * np.sin(np.pi * (8 * (x / 63) ** 2 + 2 * (y / 16) ** 2))
        values[..., 2] = 1 - values[..., 1]
    else:
        raise AssertionError(pattern)
    return values


def quality_words(pattern: str, range_token: str) -> np.ndarray:
    values = quality_values(pattern)
    y = values[..., 0]
    cb = values[:, ::2, 1]
    cr = values[:, ::2, 2]
    if pattern == "alternation":
        sample_index = np.arange(32)[None, :]
        cb = np.broadcast_to(np.where(sample_index % 2 == 0, 0.25, 0.75), (17, 32))
        cr = 1 - cb

    def quantize(plane: np.ndarray, component: int) -> np.ndarray:
        q64 = plane * 1023 if range_token == "full" else plane * (876 if component == 0 else 896) + 64
        return np.clip(np.copysign(np.floor(np.abs(q64) + 0.5), q64), 0, 1023).astype(np.uint16)

    return pack_planes(quantize(y, 0), quantize(cb, 1), quantize(cr, 2)) << 6
