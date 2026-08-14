"""Generate the manual visual-acceptance sheet for v1-unsharp-mask."""

from __future__ import annotations

import argparse
from pathlib import Path

import cupy as cp
import numpy as np

import pixtreme as px

_WIDTH = 640
_HEIGHT = 420
_LABEL_HEIGHT = 38


def _diagnostic_source() -> px.core.Frame:
    x = cp.linspace(0.0, 1.0, _WIDTH, dtype=cp.float32)[None, :]
    y = cp.linspace(0.0, 1.0, _HEIGHT, dtype=cp.float32)[:, None]
    coarse_noise = px.generate.fractal_noise(
        width=_WIDTH,
        height=_HEIGHT,
        scale=112.0,
        octaves=5,
        lacunarity=2.1,
        gain=0.55,
        seed=1977,
        evolution=0.35,
        colorspace="ACEScg",
    ).data[..., 0]
    fine_noise = px.generate.fractal_noise(
        width=_WIDTH,
        height=_HEIGHT,
        scale=19.0,
        octaves=4,
        lacunarity=2.0,
        gain=0.58,
        seed=811,
        evolution=1.25,
        colorspace="ACEScg",
    ).data[..., 0]
    micro_texture = cp.sin(np.float32(113.0) * x + np.float32(41.0) * y) * cp.sin(
        np.float32(53.0) * x - np.float32(97.0) * y
    )
    vignette = cp.maximum(
        np.float32(0.0),
        np.float32(1.0) - np.float32(1.7) * ((x - np.float32(0.5)) ** 2 + (y - np.float32(0.5)) ** 2),
    )
    data = cp.empty((_HEIGHT, _WIDTH, 3), dtype=cp.float32)
    data[..., 0] = (
        np.float32(0.02)
        + np.float32(0.34) * x
        + np.float32(0.25) * coarse_noise
        + np.float32(0.10) * fine_noise
        + np.float32(0.035) * micro_texture
    )
    data[..., 1] = (
        np.float32(0.03)
        + np.float32(0.25) * y
        + np.float32(0.18) * coarse_noise
        + np.float32(0.13) * fine_noise
        - np.float32(0.025) * micro_texture
    )
    data[..., 2] = (
        np.float32(0.04)
        + np.float32(0.30) * vignette
        + np.float32(0.11) * coarse_noise
        + np.float32(0.16) * fine_noise
        + np.float32(0.020) * micro_texture
    )

    top_edge = y < np.float32(14.0 / _HEIGHT)
    bottom_edge = y > np.float32((_HEIGHT - 15.0) / _HEIGHT)
    left_edge = x < np.float32(13.0 / _WIDTH)
    right_edge = x > np.float32((_WIDTH - 17.0) / _WIDTH)
    edge_checker = cp.mod(cp.floor(x * np.float32(32.0)), np.float32(2.0)) < np.float32(1.0)
    data[..., 0] = cp.where(top_edge, np.float32(1.30) - np.float32(0.80) * x, data[..., 0])
    data[..., 1] = cp.where(left_edge, np.float32(1.20) - np.float32(0.90) * y, data[..., 1])
    data[..., 2] = cp.where(right_edge, np.float32(0.08) + np.float32(1.20) * y, data[..., 2])
    data[..., 0] = cp.where(bottom_edge & edge_checker, np.float32(1.40), data[..., 0])
    data[..., 1] = cp.where(bottom_edge & ~edge_checker, np.float32(0.02), data[..., 1])
    data[..., 2] = cp.where(bottom_edge, np.float32(0.20) + np.float32(0.70) * x, data[..., 2])
    frame = px.io.from_array(data, colorspace="ACEScg", gamma="linear", channels="RGB")

    frame = px.draw.rectangle(
        frame,
        top_left=(24.0, 260.0),
        bottom_right=(232.0, 385.0),
        color=(0.03, 0.03, 0.03),
        fill=True,
        aa="off",
    )
    frame = px.draw.rectangle(
        frame,
        top_left=(42.0, 278.0),
        bottom_right=(214.0, 367.0),
        color=(1.5, 1.5, 1.5),
        fill=True,
        aa="off",
    )
    frame = px.draw.rectangle(
        frame,
        top_left=(258.0, 246.0),
        bottom_right=(610.0, 390.0),
        color=(0.02, 0.02, 0.02),
        fill=True,
        aa="off",
    )
    for offset in range(0, 324, 9):
        frame = px.draw.line(
            frame,
            start=(270.0 + offset, 257.0),
            end=(270.0, 380.0 - offset * 0.22),
            color=(1.15, 0.95, 0.55),
            thickness=1.0,
            aa="distance",
        )
    frame = px.draw.circle(
        frame,
        center=(520.0, 150.0),
        radius=72.0,
        color=(1.4, 0.55, 0.12),
        fill=True,
        aa="distance",
    )
    frame = px.draw.circle(
        frame,
        center=(520.0, 150.0),
        radius=42.0,
        color=(-0.15, 0.02, 0.08),
        fill=True,
        aa="distance",
    )
    frame = px.draw.text(
        frame,
        text="Pixtreme\n文字 EDGE",
        position=(34.0, 55.0),
        size=42.0,
        color=(1.25, 1.25, 1.25),
        weight=700.0,
        anchor="top-left",
        outlines=(((0.0, 0.0, 0.0), 2.0),),
        line_spacing=1.05,
    )
    frame = px.draw.text(
        frame,
        text="13 px fine text / 細字 123 ABC",
        position=(302.0, 38.0),
        size=13.0,
        color=(0.95, 0.95, 0.95),
        anchor="top-left",
        weight=400.0,
    )
    frame = px.draw.text(
        frame,
        text="MEDIUM 24 / 中サイズ",
        position=(302.0, 66.0),
        size=24.0,
        color=(0.08, 0.08, 0.08),
        anchor="top-left",
        weight=700.0,
        outlines=(((1.05, 1.05, 1.05), 1.0),),
    )
    frame = px.draw.text(
        frame,
        text="1.5 scene highlight / fractal + 1 px detail",
        position=(38.0, 404.0),
        size=16.0,
        color=(0.9, 0.9, 0.9),
        anchor="bottom-left",
    )
    return frame


def _label(frame: px.core.Frame, text: str) -> px.core.Frame:
    label_data = cp.full((_LABEL_HEIGHT, _WIDTH, 3), np.float32(0.015), dtype=cp.float32)
    label_bar = px.io.from_array(
        label_data,
        colorspace=frame.colorspace,
        gamma=frame.gamma,
        channels=frame.channels,
    )
    label_bar = px.draw.text(
        label_bar,
        text=text,
        position=(12.0, 27.0),
        size=18.0,
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
        _label(source, "INPUT (scene-linear display clip)"),
        _label(
            px.filter.unsharp_mask(source, sigma=1.2, amount=1.5, border="mirror"),
            "AMOUNT POSITIVE  sigma=1.2  amount=+1.5  border=mirror",
        ),
        _label(
            px.filter.unsharp_mask(source, sigma=1.2, amount=-1.0, border="mirror"),
            "AMOUNT NEGATIVE  sigma=1.2  amount=-1.0 (= gaussian blur)",
        ),
        _label(
            px.filter.unsharp_mask(source, sigma=1.2, amount=-0.0, border="wrap"),
            "AMOUNT ZERO  amount=-0.0  BIT-EXACT IDENTITY  border=wrap",
        ),
    )
    border_panels = (
        _label(
            px.filter.unsharp_mask(source, sigma=6.0, amount=2.0, border="mirror"),
            "BORDER mirror  sigma=6.0  amount=+2.0",
        ),
        _label(
            px.filter.unsharp_mask(source, sigma=6.0, amount=2.0, border="replicate"),
            "BORDER replicate  sigma=6.0  amount=+2.0",
        ),
        _label(
            px.filter.unsharp_mask(source, sigma=6.0, amount=2.0, border="wrap"),
            "BORDER wrap  sigma=6.0  amount=+2.0",
        ),
        _label(
            px.filter.unsharp_mask(source, sigma=6.0, amount=2.0, border="constant", border_value=-0.25),
            "BORDER constant  value=-0.25  sigma=6.0  amount=+2.0",
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
