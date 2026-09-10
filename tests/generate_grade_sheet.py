"""Generate the manual visual-acceptance sheet for v1-grade."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import cupy as cp
import numpy as np

import pixtreme as px

_WIDTH = 320
_HEIGHT = 180
_LABEL_HEIGHT = 38
_CHANNELS = ("R", "G", "B")

_Parameter = float | Mapping[str, float]
_Case = tuple[str, _Parameter, _Parameter, _Parameter]


def _source_data() -> np.ndarray:
    horizontal = np.linspace(-0.5, 2.0, _WIDTH, dtype=np.float32)
    vertical = np.linspace(0.0, 1.0, _HEIGHT, dtype=np.float32)[:, None]
    data = np.empty((_HEIGHT, _WIDTH, 3), dtype=np.float32)
    data[..., 0] = horizontal[None, :]
    data[..., 1] = np.float32(0.1) + np.float32(1.25) * vertical
    data[..., 2] = np.float32(1.5) - np.float32(0.75) * horizontal[None, :]
    patches = (
        ((12, 54, 12, 68), (-0.35, 0.18, 0.42)),
        ((12, 54, 76, 132), (0.18, 0.18, 0.18)),
        ((12, 54, 140, 196), (0.72, 0.38, 0.14)),
        ((12, 54, 204, 260), (1.0, 1.0, 1.0)),
        ((12, 54, 268, 316), (1.8, 1.25, 0.8)),
        ((68, 118, 20, 94), (1.4, -0.2, 0.1)),
        ((68, 118, 104, 178), (0.1, 1.6, 0.25)),
        ((68, 118, 188, 262), (0.15, 0.25, 1.75)),
    )
    for (top, bottom, left, right), value in patches:
        data[top:bottom, left:right] = np.asarray(value, dtype=np.float32)
    data[132:164, 18:146] = np.asarray((-0.5, -0.25, -0.0), dtype=np.float32)
    data[132:164, 174:302] = np.asarray((1.0, 1.4, 2.0), dtype=np.float32)
    return data


def _frame(data: np.ndarray) -> px.core.Frame:
    return px.io.from_array(
        cp.asarray(np.asarray(data, dtype=np.float32)),
        colorspace="ACEScg",
        gamma="linear",
        channels="RGB",
    )


def _resolved(parameter: _Parameter, neutral: float) -> np.ndarray:
    if isinstance(parameter, Mapping):
        return np.asarray([np.float32(parameter.get(label, neutral)) for label in _CHANNELS], dtype=np.float32)
    return np.full(3, np.float32(parameter), dtype=np.float32)


def _oracle(data: np.ndarray, *, lift: _Parameter, gamma: _Parameter, gain: _Parameter) -> np.ndarray:
    """Evaluate the feature equation independently in host fp64 from binary32 inputs."""
    x = np.asarray(data, dtype=np.float32).astype(np.float64)
    lift64 = _resolved(lift, 0.0).astype(np.float64)
    gamma64 = _resolved(gamma, 1.0).astype(np.float64)
    gain64 = _resolved(gain, 1.0).astype(np.float64)
    z = gain64 * x + lift64 * (np.float64(1.0) - x)
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        output = np.copysign(np.power(np.abs(z), np.float64(1.0) / gamma64), z)
    neutral = (lift64 == 0.0) & (gamma64 == 1.0) & (gain64 == 1.0)
    output[..., neutral] = x[..., neutral]
    return output.astype(np.float32)


def _display(data: np.ndarray) -> px.core.Frame:
    converted = px.color.rgb_to_rgb(
        _frame(data),
        output_colorspace="sRGB",
        output_gamma="sRGB",
    )
    clipped = cp.clip(converted.data, np.float32(0.0), np.float32(1.0))
    return px.io.from_array(clipped, colorspace="sRGB", gamma="sRGB", channels="RGB")


def _signed_overshoot_diagnostic(data: np.ndarray) -> px.core.Frame:
    # Map the documented diagnostic interval [-1, 2] linearly to display [0, 1].
    encoded = np.clip((np.asarray(data, dtype=np.float32) + np.float32(1.0)) / np.float32(3.0), 0.0, 1.0)
    return px.io.from_array(cp.asarray(encoded), colorspace="sRGB", gamma="sRGB", channels="RGB")


def _difference_diagnostic(expected: np.ndarray, actual: np.ndarray) -> px.core.Frame:
    error = np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)), axis=2)
    intensity = np.clip(error * np.float64(1_000_000.0), 0.0, 1.0).astype(np.float32)
    diagnostic = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.float32)
    diagnostic[..., 0] = intensity
    diagnostic[..., 2] = intensity * np.float32(0.5)
    return px.io.from_array(cp.asarray(diagnostic), colorspace="sRGB", gamma="sRGB", channels="RGB")


def _label(frame: px.core.Frame, text: str) -> px.core.Frame:
    bar = px.io.from_array(
        cp.full((_LABEL_HEIGHT, _WIDTH, 3), np.float32(0.012), dtype=cp.float32),
        colorspace="sRGB",
        gamma="sRGB",
        channels="RGB",
    )
    bar = px.draw.text(
        bar,
        text=text,
        position=(7.0, 26.0),
        size=11.0,
        color=(1.0, 1.0, 1.0),
        anchor="baseline-left",
        font="mono",
    )
    return px.transform.stack((bar, frame), direction="vertical")


def _row(source_data: np.ndarray, case: _Case) -> px.core.Frame:
    name, lift, gamma, gain = case
    source = _frame(source_data)
    expected = _oracle(source_data, lift=lift, gamma=gamma, gain=gain)
    actual = px.color.grade(source, lift=lift, gamma=gamma, gain=gain).data.get()
    maximum_error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
    neutral_g = (
        _resolved(lift, 0.0)[1] == np.float32(0.0)
        and _resolved(gamma, 1.0)[1] == np.float32(1.0)
        and _resolved(gain, 1.0)[1] == np.float32(1.0)
    )
    green_bits_equal = bool(
        np.array_equal(actual[..., 1].view(np.uint32), np.asarray(source_data)[..., 1].view(np.uint32))
    )
    mapping_note = f" / G neutral bits={green_bits_equal}" if neutral_g else ""
    panels = (
        _label(_display(source_data), f"{name} / INPUT / explicit sRGB display"),
        _label(_display(expected), "EXPECTED / independent NumPy fp64"),
        _label(_display(actual), "GPU / px.color.grade"),
        _label(_signed_overshoot_diagnostic(expected), "EXPECTED diag / [-1,2] -> [0,1]"),
        _label(_signed_overshoot_diagnostic(actual), "GPU diag / [-1,2] -> [0,1]"),
        _label(
            _difference_diagnostic(expected, actual),
            f"ABS DIFF x1e6 / max={maximum_error:.3g}{mapping_note}",
        ),
    )
    return px.transform.stack(panels, direction="horizontal")


def generate_sheet(path: Path) -> None:
    source_data = _source_data()
    cases: tuple[_Case, ...] = (
        ("NEUTRAL", 0.0, 1.0, 1.0),
        ("SCALAR lift=.08 gamma=.8 gain=1.1", 0.08, 0.8, 1.1),
        (
            "MAPPING R/B; G omitted",
            {"R": 0.12, "B": -0.08},
            {"R": 0.7, "B": 1.6},
            {"R": 1.25, "B": 0.85},
        ),
        ("SIGNED + OVERSHOOT pure power", -0.25, 1.8, 1.35),
    )
    sheet = px.transform.stack(tuple(_row(source_data, case) for case in cases), direction="vertical")
    code = cp.rint(cp.clip(sheet.data, 0.0, 1.0) * np.float32(255.0)).astype(cp.uint8)
    output = px.io.from_array(code, colorspace="sRGB", gamma="sRGB", channels="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    px.io.write_image(path, output, compression_level=6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="PNG output path")
    arguments = parser.parse_args()
    generate_sheet(arguments.output)


if __name__ == "__main__":
    main()
