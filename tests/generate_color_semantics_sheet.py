"""Generate the manual visual-acceptance sheet for v1-color-semantics."""

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
    checker = cp.mod(cp.floor(x / np.float32(20.0)) + cp.floor(y / np.float32(20.0)), np.float32(2.0))
    data = cp.empty((_HEIGHT, _WIDTH, 3), dtype=cp.float32)
    data[..., 0] = np.float32(-0.20) + np.float32(1.65) * horizontal
    data[..., 1] = np.float32(0.05) + np.float32(1.20) * vertical
    data[..., 2] = np.float32(0.08) + np.float32(0.65) * checker
    data[24:76, 36:112] = cp.asarray((4.0, -0.6, 0.2), dtype=cp.float32)
    data[98:158, 205:292] = cp.asarray((-1.0, 0.4, 8.0), dtype=cp.float32)
    return px.io.from_array(data, colorspace="ACEScg", gamma="linear", channels="RGB")


def _display(frame: px.core.Frame) -> px.core.Frame:
    converted = px.color.rgb_to_rgb(frame, output_colorspace="sRGB", output_gamma="srgb")
    data = cp.clip(converted.data, np.float32(0.0), np.float32(1.0))
    return px.io.from_array(data, colorspace="sRGB", gamma="srgb", channels="RGB")


def _gray_display(frame: px.core.Frame) -> px.core.Frame:
    rgb = cp.repeat(frame.data, 3, axis=2)
    return _display(px.io.from_array(rgb, colorspace=frame.colorspace, gamma=frame.gamma, channels="RGB"))


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


def _grayscale_gamma_row(source: px.core.Frame) -> px.core.Frame:
    linear_gray = px.color.rgb_to_grayscale(source, matrix="native")
    encoded_gray = px.color.rgb_to_grayscale(source, colorspace="sRGB", gamma="srgb")
    power = px.color.linear_to_gamma(source, gamma="2.6")
    raw_power = px.io.from_array(
        cp.clip(power.data, np.float32(0.0), np.float32(1.0)),
        colorspace="sRGB",
        gamma="srgb",
        channels="RGB",
    )
    return px.transform.stack(
        (
            _label(_display(source), "INPUT / ACEScg linear / display clamp"),
            _label(_gray_display(linear_gray), "GRAYSCALE linear / native luminance"),
            _label(_gray_display(encoded_gray), "GRAYSCALE sRGB / bt709 luma"),
            _label(raw_power, "GAMMA 2.6 encoded values / raw-code view"),
        ),
        direction="horizontal",
    )


def _matrix_row(source: px.core.Frame) -> px.core.Frame:
    panels = [_label(_display(source), "RGB SOURCE / matrix comparison")]
    for matrix in ("bt601", "bt709", "bt2020"):
        ycbcr = px.color.rgb_to_ycbcr(source, colorspace="sRGB", gamma="srgb", matrix=matrix)
        y = px.io.from_array(
            ycbcr.data[..., ycbcr.channels.index("Y") : ycbcr.channels.index("Y") + 1],
            colorspace=ycbcr.colorspace,
            gamma=ycbcr.gamma,
            channels="Y",
            matrix=matrix,
        )
        panels.append(_label(_gray_display(y), f"RGB->YCbCr {matrix} / Y channel"))
    return px.transform.stack(tuple(panels), direction="horizontal")


def _rendering_row(source: px.core.Frame) -> px.core.Frame:
    encoded = px.color.rgb_to_ycbcr(source, colorspace="Rec.709", gamma="rec709", matrix="bt709")
    remapped = px.color.ycbcr_to_ycbcr(
        encoded,
        colorspace="ACEScg",
        gamma="linear",
        output_matrix="native",
    )
    remapped_rgb = px.color.ycbcr_to_rgb(
        remapped,
        colorspace="sRGB",
        gamma="srgb",
        matrix="native",
    )
    aces13 = px.color.rgb_to_rgb(source, output_colorspace="sRGB", output_gamma="srgb", tonemap="aces-1.3")
    aces20 = px.color.rgb_to_rgb(source, output_colorspace="sRGB", output_gamma="srgb", tonemap="aces-2.0-lut")
    outside = cp.zeros_like(aces13.data)
    outside[..., 0] = cp.any(aces13.data > np.float32(1.0), axis=2)
    outside[..., 2] = cp.any(aces13.data < np.float32(0.0), axis=2)
    outside[..., 1] = np.float32(0.15) * cp.all(
        (aces13.data >= np.float32(0.0)) & (aces13.data <= np.float32(1.0)), axis=2
    )
    outside_frame = px.io.from_array(outside, colorspace="sRGB", gamma="srgb", channels="RGB")
    return px.transform.stack(
        (
            _label(_display(remapped_rgb), "YCbCr bt709 -> ACEScg linear native"),
            _label(_display(aces13), "TONEMAP aces-1.3 analytic / display clamp"),
            _label(_display(aces20), "TONEMAP aces-2.0-lut / display clamp"),
            _label(outside_frame, "ACES 1.3 OUTSIDE / red >1, blue <0"),
        ),
        direction="horizontal",
    )


def generate_sheet(path: Path) -> None:
    source = _source()
    sheet = px.transform.stack(
        (_grayscale_gamma_row(source), _matrix_row(source), _rendering_row(source)),
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
