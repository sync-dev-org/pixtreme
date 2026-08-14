"""Generate the manual visual-acceptance sheet for v1-derivative-filters."""

from __future__ import annotations

import argparse
from pathlib import Path

import cupy as cp
import numpy as np

import pixtreme as px

_WIDTH = 360
_HEIGHT = 240
_LABEL_HEIGHT = 36


def _diagnostic_source() -> px.core.Frame:
    y, x = cp.indices((_HEIGHT, _WIDTH), dtype=cp.float32)
    normalized_x = x / np.float32(_WIDTH - 1)
    normalized_y = y / np.float32(_HEIGHT - 1)
    checker = cp.mod(cp.floor(x / np.float32(12.0)) + cp.floor(y / np.float32(12.0)), np.float32(2.0))
    circle = (x - np.float32(260.0)) ** 2 + (y - np.float32(122.0)) ** 2 <= np.float32(58.0**2)
    diagonal = cp.abs(y - np.float32(0.55) * x - np.float32(22.0)) < np.float32(3.0)
    data = cp.empty((_HEIGHT, _WIDTH, 3), dtype=cp.float32)
    data[..., 0] = np.float32(0.08) + np.float32(0.70) * normalized_x
    data[..., 1] = np.float32(0.10) + np.float32(0.65) * normalized_y
    data[..., 2] = np.float32(0.12) + np.float32(0.22) * checker
    data[..., 0] = cp.where((x > np.float32(42.0)) & (x < np.float32(112.0)), np.float32(1.25), data[..., 0])
    data[..., 1] = cp.where((y > np.float32(54.0)) & (y < np.float32(108.0)), np.float32(-0.15), data[..., 1])
    data[..., 2] = cp.where(circle, np.float32(1.15), data[..., 2])
    data = cp.where(diagonal[..., None], np.float32(0.92), data)
    data[:, :5, 0] = np.float32(1.30)
    data[:5, :, 1] = np.float32(-0.20)
    data[-5:, :, 2] = np.float32(1.20)
    return px.io.from_array(data, colorspace="ACEScg", gamma="linear", channels="RGB")


def _display(frame: px.core.Frame, *, scale: float, offset: float) -> px.core.Frame:
    data = cp.clip(np.float32(offset) + np.float32(scale) * frame.data, np.float32(0.0), np.float32(1.0))
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
        position=(8.0, 25.0),
        size=13.0,
        color=(1.0, 1.0, 1.0),
        anchor="baseline-left",
        font="mono",
    )
    return px.transform.stack((bar, frame), direction="vertical")


def _source_panel(source: px.core.Frame) -> px.core.Frame:
    return _label(_display(source, scale=1.0, offset=0.0), "INPUT (display clamp; scene range -0.2..1.3)")


def _sobel_direction_row(source: px.core.Frame) -> px.core.Frame:
    panels = (
        _source_panel(source),
        _label(
            _display(px.filter.sobel(source, direction="x"), scale=0.08, offset=0.5), "SOBEL x / VIEW=.5+.08*response"
        ),
        _label(
            _display(px.filter.sobel(source, direction="y"), scale=0.08, offset=0.5), "SOBEL y / VIEW=.5+.08*response"
        ),
        _label(
            _display(px.filter.sobel(source, direction="magnitude"), scale=0.08, offset=0.0),
            "SOBEL magnitude / VIEW=.08*response",
        ),
        _label(_display(px.filter.sobel(source), scale=0.08, offset=0.0), "SOBEL default=magnitude"),
    )
    return px.transform.stack(panels, direction="horizontal")


def _border_row(source: px.core.Frame, *, operation: str) -> px.core.Frame:
    if operation == "sobel":
        apply = lambda border, border_value=None: px.filter.sobel(  # noqa: E731
            source, direction="magnitude", border=border, border_value=border_value
        )
        scale = 0.08
        offset = 0.0
        title = "SOBEL mag VIEW=.08*r"
    elif operation == "laplacian":
        apply = lambda border, border_value=None: px.filter.laplacian(  # noqa: E731
            source, border=border, border_value=border_value
        )
        scale = 0.20
        offset = 0.5
        title = "LAPLACIAN VIEW=.5+.2*r"
    else:
        apply = lambda border, border_value=None: px.filter.difference_of_gaussians(  # noqa: E731
            source, sigma1=1.0, sigma2=2.0, border=border, border_value=border_value
        )
        scale = 2.0
        offset = 0.5
        title = "DoG 1-2 VIEW=.5+2*r"
    panels = (
        _source_panel(source),
        _label(_display(apply("mirror"), scale=scale, offset=offset), f"{title} / mirror"),
        _label(_display(apply("replicate"), scale=scale, offset=offset), f"{title} / replicate"),
        _label(_display(apply("wrap"), scale=scale, offset=offset), f"{title} / wrap"),
        _label(_display(apply("constant", -0.4), scale=scale, offset=offset), f"{title} / constant=-.4"),
    )
    return px.transform.stack(panels, direction="horizontal")


def _dog_sigma_row(source: px.core.Frame) -> px.core.Frame:
    panels = (
        _source_panel(source),
        _label(
            _display(px.filter.difference_of_gaussians(source, sigma1=0.7, sigma2=1.4), scale=2.0, offset=0.5),
            "DoG FINE sigma1=0.7 sigma2=1.4 / signed offset",
        ),
        _label(
            _display(px.filter.difference_of_gaussians(source, sigma1=2.0, sigma2=5.0), scale=2.0, offset=0.5),
            "DoG COARSE sigma1=2.0 sigma2=5.0 / signed offset",
        ),
        _label(
            _display(px.filter.difference_of_gaussians(source, sigma1=1.4, sigma2=0.7), scale=2.0, offset=0.5),
            "DoG REVERSED 1.4-0.7 / sign inversion",
        ),
        _label(
            _display(px.filter.difference_of_gaussians(source, sigma1=1.2, sigma2=1.2), scale=2.0, offset=0.5),
            "DoG EQUAL 1.2-1.2 / zero image = neutral 0.5",
        ),
    )
    return px.transform.stack(panels, direction="horizontal")


def generate_sheet(path: Path) -> None:
    source = _diagnostic_source()
    sheet = px.transform.stack(
        (
            _sobel_direction_row(source),
            _border_row(source, operation="sobel"),
            _border_row(source, operation="laplacian"),
            _border_row(source, operation="dog"),
            _dog_sigma_row(source),
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
