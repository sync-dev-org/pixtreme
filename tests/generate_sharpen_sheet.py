"""Generate the manual visual-acceptance sheet for v1-sharpen."""

from __future__ import annotations

import argparse
from pathlib import Path

import cupy as cp
import numpy as np

import pixtreme as px

_WIDTH = 480
_HEIGHT = 300
_LABEL_HEIGHT = 42


def _diagnostic_source() -> px.core.Frame:
    x = cp.linspace(0.0, 1.0, _WIDTH, dtype=cp.float32)[None, :]
    y = cp.linspace(0.0, 1.0, _HEIGHT, dtype=cp.float32)[:, None]
    fine = cp.sin(np.float32(151.0) * x + np.float32(67.0) * y) * cp.sin(np.float32(73.0) * x - np.float32(131.0) * y)
    checker = cp.mod(cp.floor(x * np.float32(40.0)) + cp.floor(y * np.float32(24.0)), np.float32(2.0))
    radial = cp.maximum(
        np.float32(0.0),
        np.float32(1.0) - np.float32(7.0) * ((x - np.float32(0.72)) ** 2 + (y - np.float32(0.35)) ** 2),
    )

    data = cp.empty((_HEIGHT, _WIDTH, 3), dtype=cp.float32)
    data[..., 0] = np.float32(-0.10) + np.float32(0.85) * x + np.float32(0.08) * fine
    data[..., 1] = np.float32(0.04) + np.float32(0.62) * y + np.float32(0.10) * checker
    data[..., 2] = np.float32(0.02) + np.float32(1.35) * radial + np.float32(0.05) * fine

    # Strong scene-linear edges and one-pixel structures make halo polarity visible.
    data[174:275, 24:216] = np.float32(0.02)
    data[194:255, 48:192] = np.float32(1.50)
    data[205:244, 74:166] = np.float32(-0.20)
    data[35:145, 282:284] = np.float32(1.40)
    data[35:145, 300:301] = np.float32(-0.25)
    data[35:145, 316:320] = np.float32(1.15)

    # Edge-touching strips expose the four border modes in the second row.
    data[:3, :, 0] = cp.where(checker[:1] > np.float32(0.5), np.float32(1.45), np.float32(-0.15))
    data[-3:, :, 1] = cp.where(checker[:1] > np.float32(0.5), np.float32(1.30), np.float32(0.0))
    data[:, :3, 2] = np.float32(1.35) - np.float32(1.25) * y
    data[:, -3:, 0] = np.float32(-0.10) + np.float32(1.40) * y

    return px.io.from_array(data, colorspace="ACEScg", gamma="linear", channels="RGB")


def _label(frame: px.core.Frame, text: str) -> px.core.Frame:
    minimum = float(cp.min(frame.data).get())
    maximum = float(cp.max(frame.data).get())
    label_data = cp.full((_LABEL_HEIGHT, _WIDTH, 3), np.float32(0.015), dtype=cp.float32)
    label_bar = px.io.from_array(
        label_data,
        colorspace=frame.colorspace,
        gamma=frame.gamma,
        channels=frame.channels,
    )
    label_bar = px.draw.text(
        label_bar,
        text=f"{text}   min={minimum:+.3f} max={maximum:+.3f}",
        position=(10.0, 29.0),
        size=16.0,
        color=(1.0, 1.0, 1.0),
        anchor="baseline-left",
        font="mono",
    )
    return px.transform.stack((label_bar, frame), direction="vertical")


def _display_frame(frame: px.core.Frame) -> px.core.Frame:
    display = cp.rint(cp.clip(frame.data, 0.0, 1.0) * np.float32(255.0)).astype(cp.uint8)
    return px.io.from_array(display, colorspace="sRGB", gamma="srgb", channels="RGB")


def generate_sheet(path: Path) -> None:
    source = _diagnostic_source()
    amount_panels = (
        _label(source, "INPUT / scene-linear display clip"),
        _label(px.filter.sharpen(source, amount=1.0), "AMOUNT +1.0 / sharpen halo"),
        _label(px.filter.sharpen(source, amount=-1.0), "AMOUNT -1.0 / inverse response"),
        _label(px.filter.sharpen(source, amount=-0.0, border="wrap"), "AMOUNT -0.0 / bit-exact identity"),
    )
    border_panels = (
        _label(px.filter.sharpen(source, amount=3.0, border="mirror"), "BORDER mirror / amount +3"),
        _label(px.filter.sharpen(source, amount=3.0, border="replicate"), "BORDER replicate / amount +3"),
        _label(px.filter.sharpen(source, amount=3.0, border="wrap"), "BORDER wrap / amount +3"),
        _label(
            px.filter.sharpen(source, amount=3.0, border="constant", border_value=-0.5),
            "BORDER constant -0.5 / amount +3",
        ),
    )
    rows = (
        px.transform.stack(tuple(_display_frame(panel) for panel in amount_panels), direction="horizontal"),
        px.transform.stack(tuple(_display_frame(panel) for panel in border_panels), direction="horizontal"),
    )
    sheet = px.transform.stack(rows, direction="vertical")
    path.parent.mkdir(parents=True, exist_ok=True)
    px.io.write_image(path, sheet, compression_level=6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="PNG output path")
    arguments = parser.parse_args()
    generate_sheet(arguments.output)


if __name__ == "__main__":
    main()
