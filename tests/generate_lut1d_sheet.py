"""Generate the manual visual-acceptance sheet for the v1-lut-extensions 1D CUDA path."""

from __future__ import annotations

import argparse
from pathlib import Path

import cupy as cp
import numpy as np

import pixtreme as px

_WIDTH = 320
_HEIGHT = 180
_LABEL_HEIGHT = 42


def _rgba_source(*, outside_domain: bool) -> px.core.Frame:
    y, x = cp.indices((_HEIGHT, _WIDTH), dtype=cp.float32)
    horizontal = x / np.float32(_WIDTH - 1)
    vertical = y / np.float32(_HEIGHT - 1)
    checker = cp.mod(cp.floor(x / np.float32(24.0)) + cp.floor(y / np.float32(18.0)), np.float32(2.0))
    scale = np.float32(2.0) if outside_domain else np.float32(1.0)
    offset = np.float32(-0.5) if outside_domain else np.float32(0.0)
    data = cp.empty((_HEIGHT, _WIDTH, 4), dtype=cp.float32)
    data[..., 0] = offset + scale * horizontal
    data[..., 1] = offset + scale * vertical
    data[..., 2] = offset + scale * (np.float32(0.15) + np.float32(0.7) * checker)
    data[..., 3] = cp.where(
        cp.sin(x * np.float32(0.19)) * cp.cos(y * np.float32(0.23)) >= 0.0,
        np.float32(0.125),
        np.float32(0.875),
    )
    return px.io.from_array(data, colorspace="ACEScg", gamma="linear", channels="RGBA")


def _lut1d(table: tuple[tuple[float, float, float], ...]) -> px.core.Lut1D:
    return px.core.Lut1D(cp.asarray(table, dtype=cp.float32))


def _identity_lut() -> px.core.Lut1D:
    axis = cp.linspace(0.0, 1.0, 17, dtype=cp.float32)
    return px.core.Lut1D(cp.repeat(axis[:, None], 3, axis=1))


def _rgb_display(frame: px.core.Frame) -> px.core.Frame:
    indices = tuple(frame.channels.index(label) for label in "RGB")
    data = cp.clip(frame.data[..., indices], np.float32(0.0), np.float32(1.0))
    return px.io.from_array(data, colorspace="sRGB", gamma="srgb", channels="RGB")


def _rgb_range_display(frame: px.core.Frame, *, lower: float, upper: float) -> px.core.Frame:
    indices = tuple(frame.channels.index(label) for label in "RGB")
    data = (frame.data[..., indices] - np.float32(lower)) / np.float32(upper - lower)
    return px.io.from_array(cp.clip(data, 0.0, 1.0), colorspace="sRGB", gamma="srgb", channels="RGB")


def _plane(values: cp.ndarray, *, gain: float = 1.0) -> px.core.Frame:
    normalized = cp.clip(values * np.float32(gain), np.float32(0.0), np.float32(1.0))
    data = cp.repeat(normalized[..., None], 3, axis=2)
    return px.io.from_array(data, colorspace="sRGB", gamma="srgb", channels="RGB")


def _panel(frame: px.core.Frame, label: str) -> px.core.Frame:
    bar = px.io.from_array(
        cp.full((_LABEL_HEIGHT, _WIDTH, 3), np.float32(0.015), dtype=cp.float32),
        colorspace="sRGB",
        gamma="srgb",
        channels="RGB",
    )
    bar = px.draw.text(
        bar,
        text=label,
        position=(7.0, 14.0),
        size=9.0,
        color=(1.0, 1.0, 1.0),
        anchor="baseline-left",
        font="mono",
        line_spacing=1.0,
    )
    return px.transform.stack((bar, frame), direction="vertical")


def generate_sheet(path: Path) -> None:
    in_domain = _rgba_source(outside_domain=False)
    outside_domain = _rgba_source(outside_domain=True)
    identity = _identity_lut()
    curved = _lut1d(
        (
            (0.0, 1.0, 0.05),
            (0.8, 0.15, 0.95),
            (0.2, 0.85, 0.20),
            (1.0, 0.0, 0.75),
            (0.35, 0.65, 0.0),
        )
    )

    identity_output = px.color.apply_lut(in_domain, lut=identity)
    curve_output = px.color.apply_lut(in_domain, lut=curved, interpolation="linear")
    clamp_output = px.color.apply_lut(outside_domain, lut=identity)
    identity_error = cp.max(cp.abs(identity_output.data[..., :3] - in_domain.data[..., :3]), axis=2)
    alpha_input = in_domain.data[..., in_domain.channels.index("A")]
    alpha_output = curve_output.data[..., curve_output.channels.index("A")]
    alpha_bit_error = (alpha_input.view(cp.uint32) != alpha_output.view(cp.uint32)).astype(cp.float32)

    sheet = px.transform.stack(
        (
            px.transform.stack(
                (
                    _panel(_rgb_display(in_domain), "IDENTITY INPUT\nin-domain RGB + A pattern"),
                    _panel(_rgb_display(identity_output), "IDENTITY OUTPUT\nlinear 17-sample curves"),
                    _panel(
                        _plane(identity_error, gain=1048576.0), "IDENTITY ABS ERROR x1048576\nblack is exact/near-exact"
                    ),
                ),
                direction="horizontal",
            ),
            px.transform.stack(
                (
                    _panel(_rgb_display(curve_output), "NON-MONOTONIC OUTPUT\ndifferent R/G/B curves"),
                    _panel(
                        _rgb_range_display(outside_domain, lower=-0.5, upper=1.5),
                        "DOMAIN INPUT / RANGE NORMALIZED\nraw RGB range -0.5..1.5",
                    ),
                    _panel(_rgb_display(clamp_output), "DOMAIN CLAMP OUTPUT / RAW VIEW\nflat endpoint regions"),
                ),
                direction="horizontal",
            ),
            px.transform.stack(
                (
                    _panel(_plane(alpha_input), "A INPUT\n0.125 / 0.875 pattern"),
                    _panel(_plane(alpha_output), "A AFTER CURVE LUT\nbit-preserved pass-through"),
                    _panel(_plane(alpha_bit_error), "A BIT DIFFERENCE\nblack means exact"),
                ),
                direction="horizontal",
            ),
        ),
        direction="vertical",
    )
    code = cp.rint(cp.clip(sheet.data, 0.0, 1.0) * np.float32(255.0)).astype(cp.uint8)
    output = px.io.from_array(code, colorspace="sRGB", gamma="srgb", channels="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    px.io.write_image(path, output, compression_level=6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="PNG output path")
    arguments = parser.parse_args()
    generate_sheet(arguments.output)


if __name__ == "__main__":
    main()
