"""Independent P216 fixtures and fp64 oracle from v1-p216-wire-format AC 2/4/6/10.

Sampling reuses the existing test-only H.273 coordinate/weight oracle, never
production constants. Range constants below are H.273 at n=16 (256 * 8-bit codes).
"""

from __future__ import annotations

import numpy as np
from test_from_format_spec import _axis_plan as upsample_axis
from test_to_format_spec import _downsample_reference as downsample

FROM_FILTERS = ("nearest", "bilinear", "bicubic", "b-spline", "mitchell", "lanczos2", "lanczos3", "lanczos4")
TO_FILTERS = ("nearest", "bilinear", "bicubic", "area")


def asymmetric_codes() -> np.ndarray:
    # Three rows prevent accidental 4:2:0 height requirements; low bits expose MSB shifts/masks.
    return np.asarray(
        [
            0,
            1,
            4096,
            32768,
            60160,
            65535,
            65001,
            17,
            51007,
            8191,
            4103,
            61440,
            1001,
            5003,
            9007,
            13001,
            17011,
            21013,
            4096,
            61440,
            32768,
            1,
            65535,
            4097,
            51001,
            7001,
            11003,
            57007,
            30011,
            20021,
            13,
            65003,
            4109,
            61111,
            32767,
            32769,
        ],
        dtype=np.uint16,
    )


def unpack_codes(buf: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = buf[: height * width].reshape(height, width)
    uv = buf[height * width :].reshape(height, width // 2, 2)
    return y, uv[..., 0], uv[..., 1]


def pack_planes(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    return np.concatenate((y.ravel(), np.stack((cb, cr), axis=-1).ravel()))


def decode_values(codes: np.ndarray, range_token: str, component: int) -> np.ndarray:
    values = codes.astype(np.float64)
    if range_token == "full":
        return values / 65535.0
    return (values - 4096.0) / (56064.0 if component == 0 else 57344.0)


def from_reference(buf: np.ndarray, height: int, width: int, range_token: str, interpolation: str) -> np.ndarray:
    y, cb, cr = unpack_codes(buf, height, width)
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
        plane * 65535.0 if range_token == "full" else plane * (56064.0 if component == 0 else 57344.0) + 4096.0
        for component, plane in enumerate(planes)
    ]
    q64 = pack_planes(*mapped)
    rounded = np.copysign(np.floor(np.abs(q64) + 0.5), q64)
    return np.clip(rounded, 0, 65535).astype(np.uint16), q64


def fixed_corpus() -> np.ndarray:
    values = np.random.default_rng(43).uniform(-0.125, 1.125, size=(17, 64, 3)).astype(np.float32)
    values[0, :6] = [(-0.125,) * 3, (0,) * 3, (0, 0.5, 1), (0.5,) * 3, (1,) * 3, (1.125,) * 3]
    return values


def assert_corpus_codes(actual: np.ndarray, expected: np.ndarray, q64: np.ndarray) -> None:
    """AC-43-10: each case uses every Y/C code as denominator and only bounded near ties may differ."""
    assert actual.shape == expected.shape == q64.shape == (2 * 64 * 17,)
    assert actual.dtype == np.uint16
    differences = np.abs(actual.astype(np.int64) - expected.astype(np.int64))
    assert differences.max() <= 1, f"maximum code difference: {differences.max()}"
    changed = differences != 0
    assert np.count_nonzero(changed) / actual.size <= 0.02, (
        f"different codes: {np.count_nonzero(changed)}/{actual.size}"
    )
    # Nearest member of the finite set {0.5, 1.5, ..., 65534.5}; no wraparound at container ends.
    nearest_boundary = np.clip(np.floor(q64), 0, 65534) + 0.5
    assert np.all(np.abs(q64[changed] - nearest_boundary[changed]) <= 0.125), q64[changed]
