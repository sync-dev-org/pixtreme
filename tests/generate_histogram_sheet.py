"""Generate the manual visual-acceptance sheet for v1-histogram."""

from __future__ import annotations

import argparse
from pathlib import Path

import cupy as cp
import numpy as np

import pixtreme as px

_WIDTH = 420
_HEIGHT = 260
_LABEL_HEIGHT = 44


def _diagnostic_source() -> px.core.Frame:
    y, x = cp.indices((_HEIGHT, _WIDTH), dtype=cp.float32)
    normalized_x = x / np.float32(_WIDTH - 1)
    normalized_y = y / np.float32(_HEIGHT - 1)
    fine = cp.sin(np.float32(147.0) * normalized_x + np.float32(83.0) * normalized_y)
    weave = cp.sin(np.float32(233.0) * normalized_x) * cp.cos(np.float32(191.0) * normalized_y)
    checker = cp.mod(cp.floor(x / np.float32(7.0)) + cp.floor(y / np.float32(7.0)), np.float32(2.0))
    radial = cp.exp(
        -np.float32(18.0) * ((normalized_x - np.float32(0.72)) ** 2 + (normalized_y - np.float32(0.42)) ** 2)
    )

    data = cp.empty((_HEIGHT, _WIDTH, 3), dtype=cp.float32)
    data[..., 0] = np.float32(0.16) + np.float32(0.30) * normalized_x + np.float32(0.025) * fine
    data[..., 1] = np.float32(0.18) + np.float32(0.26) * normalized_y + np.float32(0.035) * weave
    data[..., 2] = (
        np.float32(0.14)
        + np.float32(0.18) * normalized_x
        + np.float32(0.12) * normalized_y
        + np.float32(0.025) * checker
    )

    # Local low-contrast texture, fine lines, and flat ramps expose detail recovery and banding.
    texture_region = (x > np.float32(24.0)) & (x < np.float32(190.0)) & (y > np.float32(138.0))
    data[..., 0] = cp.where(texture_region, data[..., 0] + np.float32(0.055) * weave, data[..., 0])
    data[..., 1] = cp.where(texture_region, data[..., 1] + np.float32(0.045) * fine, data[..., 1])
    data[..., 2] = cp.where(texture_region, data[..., 2] + np.float32(0.040) * checker, data[..., 2])
    data[32:118, 248:250, :] = np.float32(0.62)
    data[32:118, 265:266, :] = np.float32(0.31)
    data[32:118, 282:286, :] = np.float32(0.52)

    # Unequal channel distributions and scene values make color shift and domain clamping visible.
    highlight = radial > np.float32(0.58)
    data[..., 0] = cp.where(highlight, np.float32(1.38), data[..., 0])
    data[..., 1] = cp.where(highlight, np.float32(1.12) + np.float32(0.10) * fine, data[..., 1])
    data[..., 2] = cp.where(highlight, np.float32(0.76), data[..., 2])
    data[178:238, 300:394, 0] = np.float32(-0.22)
    data[178:238, 300:394, 1] = np.float32(0.08) + np.float32(0.18) * normalized_y[178:238, 300:394]
    data[178:238, 300:394, 2] = np.float32(1.28)
    return px.io.from_array(data, colorspace="ACEScg", gamma="linear", channels="RGB")


def _display(frame: px.core.Frame) -> px.core.Frame:
    return px.io.from_array(
        cp.clip(frame.data, np.float32(0.0), np.float32(1.0)),
        colorspace="sRGB",
        gamma="srgb",
        channels="RGB",
    )


def _label(frame: px.core.Frame, text: str) -> px.core.Frame:
    minimum = float(cp.min(frame.data).get())
    maximum = float(cp.max(frame.data).get())
    bar = px.io.from_array(
        cp.full((_LABEL_HEIGHT, _WIDTH, 3), np.float32(0.015), dtype=cp.float32),
        colorspace="sRGB",
        gamma="srgb",
        channels="RGB",
    )
    bar = px.draw.text(
        bar,
        text=f"{text}  min={minimum:+.3f} max={maximum:+.3f}",
        position=(8.0, 29.0),
        size=14.0,
        color=(1.0, 1.0, 1.0),
        anchor="baseline-left",
        font="mono",
    )
    return px.transform.stack((bar, _display(frame)), direction="vertical")


def _rows(source: px.core.Frame) -> tuple[px.core.Frame, ...]:
    overview = (
        _label(source, "INPUT / display clip / scene values"),
        _label(px.color.equalize_histogram(source), "EQUAL domain=(0,1) bins=1024"),
        _label(px.color.clahe(source), "CLAHE clip=2 tiles_y=8 tiles_x=8 bins=1024"),
        _label(
            px.color.clahe(source, domain=(-0.25, 1.5)),
            "CLAHE domain=(-.25,1.5) clip=2",
        ),
    )
    bin_comparison = (
        _label(px.color.equalize_histogram(source, bins=16), "EQUAL bins=16 / banding probe"),
        _label(px.color.equalize_histogram(source, bins=64), "EQUAL bins=64"),
        _label(px.color.equalize_histogram(source, bins=1024), "EQUAL bins=1024 default"),
        _label(
            px.color.equalize_histogram(source, domain=(-0.25, 1.5), bins=1024),
            "EQUAL domain=(-.25,1.5) bins=1024",
        ),
    )
    clip_comparison = (
        _label(px.color.clahe(source, clip_limit=1.0, bins=256), "CLAHE clip=1 bins=256"),
        _label(px.color.clahe(source, clip_limit=1.5, bins=256), "CLAHE clip=1.5 bins=256"),
        _label(px.color.clahe(source, clip_limit=2.0, bins=1024), "CLAHE clip=2 bins=1024"),
        _label(px.color.clahe(source, clip_limit=4.0, bins=1024), "CLAHE clip=4 bins=1024"),
    )
    return tuple(
        px.transform.stack(panels, direction="horizontal") for panels in (overview, bin_comparison, clip_comparison)
    )


def generate_sheet(path: Path) -> None:
    sheet = px.transform.stack(_rows(_diagnostic_source()), direction="vertical")
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
