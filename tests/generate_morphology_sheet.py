"""Generate the manual visual-acceptance sheet for v1-morphology."""

from __future__ import annotations

import argparse
from pathlib import Path

import cupy as cp
import numpy as np

import pixtreme as px

_WIDTH = 280
_HEIGHT = 180
_LABEL_HEIGHT = 34
_RADIUS = 5
_OPERATIONS = (
    ("ERODE", px.morphology.erosion),
    ("DILATE", px.morphology.dilation),
    ("OPEN", px.morphology.opening),
    ("CLOSE", px.morphology.closing),
    ("GRADIENT", px.morphology.morphological_gradient),
    ("TOPHAT", px.morphology.white_tophat),
    ("BLACKHAT", px.morphology.black_tophat),
)


def _matte_source() -> px.core.Frame:
    y, x = cp.indices((_HEIGHT, _WIDTH), dtype=cp.float32)
    disk = (x - np.float32(82.0)) ** 2 + (y - np.float32(88.0)) ** 2 <= np.float32(54.0**2)
    diamond = cp.abs(x - np.float32(194.0)) + cp.abs(y - np.float32(91.0)) <= np.float32(48.0)
    thin_cross = (cp.abs(x - np.float32(150.0)) <= np.float32(2.0)) | (cp.abs(y - np.float32(36.0)) <= np.float32(2.0))
    pinholes = ((x - np.float32(68.0)) ** 2 + (y - np.float32(74.0)) ** 2 <= np.float32(6.0**2)) | (
        (x - np.float32(96.0)) ** 2 + (y - np.float32(105.0)) ** 2 <= np.float32(3.0**2)
    )
    data = cp.where(disk | diamond | thin_cross, np.float32(0.9), np.float32(0.05))
    data = cp.where(pinholes, np.float32(0.05), data)
    data = cp.where((x < np.float32(10.0)) & (y > np.float32(112.0)), np.float32(0.72), data)
    return px.io.from_array(data[..., None], colorspace="ACEScg", gamma="linear", channels=["matte"])


def _colour_source() -> px.core.Frame:
    y, x = cp.indices((_HEIGHT, _WIDTH), dtype=cp.float32)
    normalized_x = x / np.float32(_WIDTH - 1)
    normalized_y = y / np.float32(_HEIGHT - 1)
    rings = cp.sin(cp.hypot(x - np.float32(140.0), y - np.float32(90.0)) * np.float32(0.21))
    checker = cp.mod(cp.floor(x / np.float32(17.0)) + cp.floor(y / np.float32(13.0)), np.float32(2.0))
    data = cp.empty((_HEIGHT, _WIDTH, 3), dtype=cp.float32)
    data[..., 0] = np.float32(0.12) + np.float32(0.78) * normalized_x + np.float32(0.12) * rings
    data[..., 1] = np.float32(0.08) + np.float32(0.72) * normalized_y + np.float32(0.16) * checker
    data[..., 2] = (
        np.float32(0.10)
        + np.float32(0.36) * (np.float32(1.0) - normalized_x)
        + np.float32(0.25) * (rings > np.float32(0.4))
    )
    data[:, :8, 0] = np.float32(1.0)
    data[:8, :, 1] = np.float32(0.0)
    data[-8:, :, 2] = np.float32(0.95)
    return px.io.from_array(data, colorspace="ACEScg", gamma="linear", channels="RGB")


def _display(frame: px.core.Frame) -> px.core.Frame:
    data = frame.data
    if data.shape[2] == 1:
        data = cp.repeat(data, 3, axis=2)
    return px.io.from_array(cp.clip(data, 0.0, 1.0), colorspace="sRGB", gamma="srgb", channels="RGB")


def _label(frame: px.core.Frame, text: str) -> px.core.Frame:
    display = _display(frame)
    label_data = cp.full((_LABEL_HEIGHT, _WIDTH, 3), np.float32(0.015), dtype=cp.float32)
    bar = px.io.from_array(label_data, colorspace="sRGB", gamma="srgb", channels="RGB")
    bar = px.draw.text(
        bar,
        text=text,
        position=(8.0, 24.0),
        size=14.0,
        color=(1.0, 1.0, 1.0),
        anchor="baseline-left",
        font="mono",
    )
    return px.transform.stack((bar, display), direction="vertical")


def _operation_row(source: px.core.Frame, material: str) -> px.core.Frame:
    panels = [_label(source, f"{material} INPUT")]
    panels.extend(
        _label(operation(source, radius=_RADIUS), f"{material} {name} r={_RADIUS} disk")
        for name, operation in _OPERATIONS
    )
    return px.transform.stack(tuple(panels), direction="horizontal")


def _diagnostic_row(source: px.core.Frame) -> px.core.Frame:
    uniform = px.io.from_array(
        cp.ones((_HEIGHT, _WIDTH, 1), dtype=cp.float32),
        colorspace="ACEScg",
        gamma="linear",
        channels=["matte"],
    )
    panels = (
        _label(px.morphology.erosion(source, radius=12, shape="disk"), "SHAPE erode disk r=12"),
        _label(px.morphology.erosion(source, radius=12, shape="square"), "SHAPE erode square r=12"),
        _label(px.morphology.dilation(source, radius=12, shape="disk"), "SHAPE dilate disk r=12"),
        _label(px.morphology.dilation(source, radius=12, shape="square"), "SHAPE dilate square r=12"),
        _label(px.morphology.erosion(uniform, radius=16, border="mirror"), "BORDER mirror / uniform matte"),
        _label(px.morphology.erosion(uniform, radius=16, border="replicate"), "BORDER replicate / neutral edge"),
        _label(px.morphology.erosion(uniform, radius=16, border="wrap"), "BORDER wrap / uniform matte"),
        _label(
            px.morphology.erosion(uniform, radius=16, border="constant", border_value=0.0),
            "BORDER constant=0 / edge shrinks",
        ),
    )
    return px.transform.stack(panels, direction="horizontal")


def generate_sheet(path: Path) -> None:
    matte = _matte_source()
    colour = _colour_source()
    sheet = px.transform.stack(
        (_operation_row(matte, "1CH"), _operation_row(colour, "RGB"), _diagnostic_row(matte)),
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
