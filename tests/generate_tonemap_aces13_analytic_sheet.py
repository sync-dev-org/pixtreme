"""Generate the manual visual sheet for v1-tonemap-aces13-analytic acceptance 22."""

from __future__ import annotations

import argparse
from pathlib import Path

import cupy as cp
import numpy as np

import pixtreme as px

_WIDTH = 320
_HEIGHT = 180
_LABEL_HEIGHT = 38
_SUPPLIED_COMBINATIONS = (
    ("aces-1.3", "Rec.709", "bt1886"),
    ("aces-1.3", "sRGB", "srgb"),
    ("aces-1.3-lut", "Rec.709", "bt1886"),
    ("aces-1.3-lut", "sRGB", "srgb"),
    ("aces-2.0-lut", "Rec.709", "bt1886"),
    ("aces-2.0-lut", "sRGB", "srgb"),
    ("bt2408", "Rec.2020", "hlg"),
    ("bt2408", "Rec.2020", "pq"),
)


def _source() -> px.core.Frame:
    y, x = cp.indices((_HEIGHT, _WIDTH), dtype=cp.float32)
    horizontal = x / np.float32(_WIDTH - 1)
    vertical = y / np.float32(_HEIGHT - 1)
    data = cp.empty((_HEIGHT, _WIDTH, 3), dtype=cp.float32)

    # Smooth neutral/RGB ramps expose banding and analytic piece boundaries.
    neutral = np.float32(-0.5) + np.float32(16.5) * horizontal
    data[..., 0] = neutral
    data[..., 1] = neutral * (np.float32(0.35) + np.float32(0.65) * vertical)
    data[..., 2] = neutral * (np.float32(1.0) - np.float32(0.75) * vertical)

    # Primary, secondary, skin, middle-gray, white, superwhite, and out-of-gamut patches.
    patches = cp.asarray(
        (
            (-0.5, -0.5, -0.5),
            (0.0, 0.0, 0.0),
            (0.18, 0.18, 0.18),
            (0.40, 0.18, 0.10),
            (1.0, 1.0, 1.0),
            (4.0, 4.0, 4.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 1.0),
            (1.0, 0.0, 1.0),
            (16.0, 2.0, 0.1),
            (1.2, -0.1, 0.4),
        ),
        dtype=cp.float32,
    )
    patch_width = _WIDTH // 7
    patch_height = 32
    for row in range(2):
        for column in range(7):
            left = column * patch_width
            right = _WIDTH if column == 6 else (column + 1) * patch_width
            data[row * patch_height : (row + 1) * patch_height, left:right] = patches[row * 7 + column]
    return px.io.from_array(data, colorspace="ACES2065-1", gamma="linear", channels="RGB")


def _render(source: px.core.Frame, tonemap: str, output_colorspace: str, output_gamma: str) -> px.core.Frame:
    return px.color.rgb_to_rgb(
        source,
        output_colorspace=output_colorspace,
        output_gamma=output_gamma,
        tonemap=tonemap,
    )


def _preview(frame: px.core.Frame) -> px.core.Frame:
    if frame.colorspace == "sRGB" and frame.gamma == "srgb":
        display = frame
    else:
        display = px.color.rgb_to_rgb(frame, output_colorspace="sRGB", output_gamma="srgb")
    return px.io.from_array(
        cp.clip(display.data, np.float32(0.0), np.float32(1.0)),
        colorspace="sRGB",
        gamma="srgb",
        channels="RGB",
    )


def _absolute_difference(analytic: px.core.Frame, lut: px.core.Frame) -> px.core.Frame:
    difference = cp.abs(analytic.data - lut.data)
    expanded = cp.empty_like(difference)
    maximum = cp.max(difference, axis=2)
    expanded[..., 0] = cp.clip(maximum * np.float32(80.0), 0.0, 1.0)
    expanded[..., 1] = cp.clip(maximum * np.float32(20.0), 0.0, 1.0)
    expanded[..., 2] = cp.clip(maximum * np.float32(5.0), 0.0, 1.0)
    return px.io.from_array(expanded, colorspace="sRGB", gamma="srgb", channels="RGB")


def _gamut_highlight_diagnostic(frame: px.core.Frame) -> px.core.Frame:
    minimum = cp.min(frame.data, axis=2)
    maximum = cp.max(frame.data, axis=2)
    diagnostic = cp.empty_like(frame.data)
    diagnostic[..., 0] = cp.clip(-minimum * np.float32(4.0), 0.0, 1.0)
    diagnostic[..., 1] = cp.clip(cp.mean(frame.data, axis=2), 0.0, 1.0)
    diagnostic[..., 2] = cp.clip((maximum - np.float32(1.0)) * np.float32(2.0), 0.0, 1.0)
    return px.io.from_array(diagnostic, colorspace="sRGB", gamma="srgb", channels="RGB")


def _panel(frame: px.core.Frame, label: str, *, measured: px.core.Frame) -> px.core.Frame:
    minimum = float(cp.min(measured.data).get())
    maximum = float(cp.max(measured.data).get())
    bar = px.io.from_array(
        cp.full((_LABEL_HEIGHT, _WIDTH, 3), np.float32(0.015), dtype=cp.float32),
        colorspace="sRGB",
        gamma="srgb",
        channels="RGB",
    )
    bar = px.draw.text(
        bar,
        text=f"{label}\nraw {minimum:.5f}..{maximum:.5f}",
        position=(5.0, 12.0),
        size=8.0,
        color=(1.0, 1.0, 1.0),
        anchor="baseline-left",
        font="mono",
        line_spacing=0.9,
    )
    return px.transform.stack((bar, frame), direction="vertical")


def _overview_rows(source: px.core.Frame) -> tuple[px.core.Frame, px.core.Frame]:
    panels = []
    for tonemap, output_colorspace, output_gamma in _SUPPLIED_COMBINATIONS:
        rendered = _render(source, tonemap, output_colorspace, output_gamma)
        panels.append(
            _panel(
                _preview(rendered),
                f"{tonemap} / {output_colorspace} {output_gamma}",
                measured=rendered,
            )
        )
    return (
        px.transform.stack(tuple(panels[:4]), direction="horizontal"),
        px.transform.stack(tuple(panels[4:]), direction="horizontal"),
    )


def _comparison_row(source: px.core.Frame, output_colorspace: str, output_gamma: str) -> px.core.Frame:
    analytic = _render(source, "aces-1.3", output_colorspace, output_gamma)
    lut = _render(source, "aces-1.3-lut", output_colorspace, output_gamma)
    difference = _absolute_difference(analytic, lut)
    diagnostic = _gamut_highlight_diagnostic(analytic)
    exit_label = f"{output_colorspace} {output_gamma}"
    return px.transform.stack(
        (
            _panel(_preview(analytic), f"analytic side-by-side / {exit_label}", measured=analytic),
            _panel(_preview(lut), f"aces-1.3-lut side-by-side / {exit_label}", measured=lut),
            _panel(difference, f"expanded absolute difference / {exit_label}", measured=difference),
            _panel(diagnostic, f"gamut / highlight boundary / {exit_label}", measured=analytic),
        ),
        direction="horizontal",
    )


def generate_sheet(path: Path) -> None:
    source = _source()
    overview = _overview_rows(source)
    sheet = px.transform.stack(
        (
            *overview,
            _comparison_row(source, "Rec.709", "bt1886"),
            _comparison_row(source, "sRGB", "srgb"),
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
