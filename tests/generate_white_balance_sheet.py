"""Generate the manual visual-acceptance sheet for v1-white-balance."""

from __future__ import annotations

import argparse
from pathlib import Path

import cupy as cp
import numpy as np

import pixtreme as px

_WIDTH = 320
_HEIGHT = 180
_LABEL_HEIGHT = 34

_SOURCE_WHITE = (0.34567, 0.35850)
_TARGET_WHITE = (0.31270, 0.32900)


def _source() -> px.core.Frame:
    y, x = cp.indices((_HEIGHT, _WIDTH), dtype=cp.float32)
    horizontal = x / np.float32(_WIDTH - 1)
    data = cp.empty((_HEIGHT, _WIDTH, 3), dtype=cp.float32)
    ramp = cp.broadcast_to(horizontal, (_HEIGHT, _WIDTH))
    data[..., 0] = ramp
    data[..., 1] = ramp
    data[..., 2] = ramp
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
    return px.io.from_array(data, colorspace="sRGB", gamma="sRGB", channels="RGB")


def _display(frame: px.core.Frame) -> px.core.Frame:
    data = cp.clip(frame.data, np.float32(0.0), np.float32(1.0))
    return px.io.from_array(data, colorspace="sRGB", gamma="sRGB", channels="RGB")


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
        position=(7.0, 24.0),
        size=12.0,
        color=(1.0, 1.0, 1.0),
        anchor="baseline-left",
        font="mono",
    )
    return px.transform.stack((bar, frame), direction="vertical")


def _cat_row(source: px.core.Frame) -> px.core.Frame:
    panels = [_label(_display(source), "INPUT / sRGB / display clamp")]
    for cat in ("Bradford", "CAT02", "CAT16", "von-Kries"):
        adapted = px.color.chromatic_adaptation(
            source,
            input_white=_SOURCE_WHITE,
            output_white=_TARGET_WHITE,
            cat=cat,
        )
        panels.append(_label(_display(adapted), f"CAT {cat} / D50-ish -> D65"))
    return px.transform.stack(tuple(panels), direction="horizontal")


def _temperature_row(source: px.core.Frame) -> px.core.Frame:
    panels = [_label(_display(source), "INPUT / sRGB / display clamp")]
    variants = (
        (3200.0, "low / tungsten"),
        (4500.0, "mixed"),
        (6500.0, "near nominal"),
        (12000.0, "high / blue sky"),
    )
    for kelvin, note in variants:
        balanced = px.color.white_balance(source, temperature=kelvin, tint=0.0)
        panels.append(_label(_display(balanced), f"TEMP {kelvin:g}K tint 0 / {note}"))
    return px.transform.stack(tuple(panels), direction="horizontal")


def _tint_row(source: px.core.Frame) -> px.core.Frame:
    panels = [_label(_display(source), "INPUT / sRGB / display clamp")]
    variants = (
        (0.02, "green source -> magenta out"),
        (0.01, "green source -> magenta out"),
        (-0.01, "magenta source -> green out"),
        (-0.02, "magenta source -> green out"),
    )
    for duv, note in variants:
        balanced = px.color.white_balance(source, temperature=6500.0, tint=duv)
        panels.append(_label(_display(balanced), f"TINT Duv {duv:+g} @6500K / {note}"))
    return px.transform.stack(tuple(panels), direction="horizontal")


def generate_sheet(path: Path) -> None:
    source = _source()
    sheet = px.transform.stack(
        (_cat_row(source), _temperature_row(source), _tint_row(source)),
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
