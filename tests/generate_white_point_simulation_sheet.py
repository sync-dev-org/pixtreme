"""Generate the manual visual-acceptance sheet for v1-white-point-simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

import cupy as cp
import numpy as np

import pixtreme as px

_WIDTH = 320
_HEIGHT = 180
_LABEL_HEIGHT = 34


def _source() -> px.core.Frame:
    y, x = cp.indices((_HEIGHT, _WIDTH), dtype=cp.float32)
    horizontal = x / np.float32(_WIDTH - 1)
    vertical = y / np.float32(_HEIGHT - 1)
    data = cp.empty((_HEIGHT, _WIDTH, 3), dtype=cp.float32)
    data[..., 0] = horizontal * np.float32(1.4) - np.float32(0.2)
    data[..., 1] = vertical * np.float32(1.4) - np.float32(0.2)
    data[..., 2] = (horizontal + vertical) * np.float32(0.7) - np.float32(0.2)
    patches = (
        ((12, 66, 16, 88), (0.86, 0.62, 0.52)),
        ((12, 66, 96, 168), (0.35, 0.52, 0.78)),
        ((12, 66, 176, 248), (0.22, 0.46, 0.18)),
        ((12, 66, 256, 304), (0.78, 0.16, 0.14)),
        ((78, 128, 16, 88), (1.00, 0.84, 0.12)),
        ((78, 128, 96, 168), (0.10, 0.62, 0.66)),
        ((78, 128, 176, 248), (0.52, 0.24, 0.58)),
        ((78, 128, 256, 304), (0.94, 0.94, 0.94)),
    )
    for (top, bottom, left, right), rgb in patches:
        data[top:bottom, left:right] = cp.asarray(rgb, dtype=cp.float32)
    data[140:172, 16:120] = cp.asarray((1.9, 1.4, 1.1), dtype=cp.float32)
    data[140:172, 136:240] = cp.asarray((-0.25, 0.05, -0.15), dtype=cp.float32)
    return px.io.from_array(data, colorspace="sRGB", gamma="srgb", channels="RGB")


def _display(frame: px.core.Frame) -> px.core.Frame:
    display = px.color.rgb_to_rgb(frame, output_colorspace="sRGB", output_gamma="srgb")
    data = cp.clip(display.data, np.float32(0.0), np.float32(1.0))
    return px.io.from_array(data, colorspace="sRGB", gamma="srgb", channels="RGB")


def _label(frame: px.core.Frame, text: str) -> px.core.Frame:
    bar = px.io.from_array(
        cp.full((_LABEL_HEIGHT, _WIDTH, 3), np.float32(0.015), dtype=cp.float32),
        colorspace="sRGB",
        gamma="srgb",
        channels="RGB",
    )
    bar = px.draw.text(
        bar,
        text=text,
        position=(7.0, 24.0),
        size=12.0,
        color=(1.0, 1.0, 1.0),
        anchor="baseline-left",
        font="mono",
    )
    return px.transform.stack((bar, frame), direction="vertical")


def generate_sheet(path: Path) -> None:
    source = _source()
    variants = (
        (source, "INPUT / explicit sRGB display transform"),
        (
            px.color.white_point_simulation(source, input_white="d65", output_white="d93"),
            "TOKEN d65 -> d93",
        ),
        (
            px.color.white_point_simulation(source, input_white="d93", output_white="d65"),
            "TOKEN d93 -> d65",
        ),
        (
            px.color.white_point_simulation(source, input_white="d65", output_white=(0.3127, 0.3290)),
            "IDENTITY token d65 -> xy D65",
        ),
        (
            px.color.white_point_simulation(
                source,
                input_white=(0.3127, 0.3290),
                output_white=(0.2831, 0.2971),
            ),
            "XY D65 -> D93",
        ),
    )
    panels = tuple(_label(_display(frame), label) for frame, label in variants)
    sheet = px.transform.stack(panels, direction="horizontal")
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
