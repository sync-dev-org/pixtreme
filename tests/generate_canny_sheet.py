"""Generate the manual visual-acceptance sheet for v1-canny."""

from __future__ import annotations

import argparse
from pathlib import Path

import cupy as cp
import numpy as np

import pixtreme as px

_WIDTH = 360
_HEIGHT = 220
_LABEL_HEIGHT = 38


def _diagnostic_source() -> px.core.Frame:
    y, x = cp.indices((_HEIGHT, _WIDTH), dtype=cp.float32)
    normalized_x = x / np.float32(_WIDTH - 1)
    normalized_y = y / np.float32(_HEIGHT - 1)
    checker = cp.mod(cp.floor(x / np.float32(13.0)) + cp.floor(y / np.float32(11.0)), np.float32(2.0))
    circle = (x - np.float32(275.0)) ** 2 + (y - np.float32(116.0)) ** 2 <= np.float32(58.0**2)
    diagonal = cp.abs(y - np.float32(0.48) * x - np.float32(24.0)) < np.float32(3.0)

    data = cp.empty((_HEIGHT, _WIDTH, 3), dtype=cp.float32)
    data[..., 0] = np.float32(-0.12) + np.float32(0.9) * normalized_x
    data[..., 1] = np.float32(0.04) + np.float32(0.72) * normalized_y
    data[..., 2] = np.float32(0.08) + np.float32(0.25) * checker
    data[..., 0] = cp.where((x > np.float32(42.0)) & (x < np.float32(116.0)), np.float32(1.35), data[..., 0])
    data[..., 1] = cp.where((y > np.float32(62.0)) & (y < np.float32(126.0)), np.float32(-0.25), data[..., 1])
    data[..., 2] = cp.where(circle, np.float32(1.2), data[..., 2])
    data = cp.where(diagonal[..., None], np.float32(0.92), data)

    data[:, :4, 0] = np.float32(1.35)
    data[:4, :, 1] = np.float32(-0.3)
    data[-4:, :, 2] = np.float32(1.25)
    data[:, -4:, 0] = np.float32(0.15) + np.float32(1.1) * normalized_y[:, :1]
    return px.io.from_array(data, colorspace="ACEScg", gamma="linear", channels="RGB")


def _connectivity_source() -> px.core.Frame:
    data = cp.zeros((_HEIGHT, _WIDTH, 3), dtype=cp.float32)
    data[:, _WIDTH // 2 :, 0] = np.float32(0.30)
    data[:10, _WIDTH // 2 :, 0] = np.float32(1.10)
    data[:, 72:132, 1] = np.float32(0.30)
    data[54:166, 250:, 2] = np.float32(0.30)
    data[102:118, 250:, 2] = np.float32(1.10)
    return px.io.from_array(data, colorspace="ACEScg", gamma="linear", channels="RGB")


def _display(frame: px.core.Frame) -> px.core.Frame:
    data = cp.clip(frame.data, np.float32(0.0), np.float32(1.0))
    return px.io.from_array(data, colorspace="sRGB", gamma="sRGB", channels="RGB")


def _channel_display(frame: px.core.Frame, channel: int) -> px.core.Frame:
    data = cp.repeat(frame.data[..., channel : channel + 1], 3, axis=2)
    return _display(px.io.from_array(data, colorspace="sRGB", gamma="linear", channels="RGB"))


def _label(frame: px.core.Frame, text: str) -> px.core.Frame:
    bar = px.io.from_array(
        cp.full((_LABEL_HEIGHT, _WIDTH, 3), np.float32(0.015), dtype=cp.float32),
        colorspace="sRGB",
        gamma="sRGB",
        channels="RGB",
    )
    bar = px.draw.text(
        bar,
        text=text,
        position=(8.0, 27.0),
        size=14.0,
        color=(1.0, 1.0, 1.0),
        anchor="baseline-left",
        font="mono",
    )
    return px.transform.stack((bar, frame), direction="vertical")


def _threshold_row(source: px.core.Frame) -> px.core.Frame:
    panels = (
        _label(_display(source), "INPUT / scene values display-clipped"),
        _label(_display(px.filter.canny(source, threshold_low=0.25, threshold_high=0.75)), "LOW=.25 HIGH=.75"),
        _label(_display(px.filter.canny(source, threshold_low=0.75, threshold_high=2.0)), "LOW=.75 HIGH=2.0"),
        _label(
            _display(px.filter.canny(source, threshold_low=1.0, threshold_high=1.0)), "EQUAL=1.0 / single threshold"
        ),
        _label(_display(px.filter.canny(source, threshold_low=2.0, threshold_high=4.0)), "LOW=2.0 HIGH=4.0"),
    )
    return px.transform.stack(panels, direction="horizontal")


def _connectivity_row(source: px.core.Frame) -> px.core.Frame:
    edges = px.filter.canny(source, threshold_low=0.5, threshold_high=3.0, border="replicate")
    panels = (
        _label(_display(source), "CONNECTIVITY INPUT / RGB independent"),
        _label(_display(edges), "RGB EDGES / low=.5 high=3.0"),
        _label(_channel_display(edges, 0), "R / long weak path reaches strong"),
        _label(_channel_display(edges, 1), "G / isolated weak edge rejected"),
        _label(_channel_display(edges, 2), "B / local strong + weak component"),
    )
    return px.transform.stack(panels, direction="horizontal")


def _border_row(source: px.core.Frame) -> px.core.Frame:
    panels = (
        _label(_display(source), "EDGE-TOUCHING INPUT"),
        _label(_display(px.filter.canny(source, threshold_low=0.5, threshold_high=1.5)), "BORDER mirror (default)"),
        _label(
            _display(px.filter.canny(source, threshold_low=0.5, threshold_high=1.5, border="replicate")),
            "BORDER replicate",
        ),
        _label(
            _display(px.filter.canny(source, threshold_low=0.5, threshold_high=1.5, border="wrap")),
            "BORDER wrap",
        ),
        _label(
            _display(
                px.filter.canny(
                    source,
                    threshold_low=0.5,
                    threshold_high=1.5,
                    border="constant",
                    border_value=-0.5,
                )
            ),
            "BORDER constant=-0.5",
        ),
    )
    return px.transform.stack(panels, direction="horizontal")


def generate_sheet(path: Path) -> None:
    source = _diagnostic_source()
    sheet = px.transform.stack(
        (_threshold_row(source), _connectivity_row(_connectivity_source()), _border_row(source)),
        direction="vertical",
    )
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
