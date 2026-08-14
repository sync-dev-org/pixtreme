"""Generate the manual visual-acceptance sheet for v1-hsv."""

from __future__ import annotations

import argparse
from pathlib import Path

import cupy as cp
import numpy as np

import pixtreme as px

_WIDTH = 320
_HEIGHT = 180
_LABEL_HEIGHT = 38
_ERROR_GAIN = np.float32(262144.0)


def _hsv_frame(data: cp.ndarray) -> px.core.Frame:
    return px.io.from_array(data, colorspace="ACEScg", gamma="linear", channels="HSV")


def _rgb_frame(data: cp.ndarray) -> px.core.Frame:
    return px.io.from_array(data, colorspace="ACEScg", gamma="linear", channels="RGB")


def _diagnostic_chart() -> px.core.Frame:
    y, x = cp.indices((_HEIGHT, _WIDTH), dtype=cp.float32)
    horizontal = x / np.float32(_WIDTH - 1)
    data = cp.empty((_HEIGHT, _WIDTH, 3), dtype=cp.float32)

    # Primary / secondary blocks with V=2 exercise all exact sector boundaries.
    data[..., 0] = horizontal
    data[..., 1] = np.float32(1.0)
    data[..., 2] = np.float32(2.0)
    sector_width = _WIDTH / 6.0
    data[:30, :, 0] = cp.floor(x[:30] / np.float32(sector_width)) / np.float32(6.0)

    # Hue seam crosses H=1 continuously, while the middle band sweeps saturation.
    data[30:60, :, 0] = np.float32(0.94) + np.float32(0.12) * horizontal[30:60]
    data[60:120, :, 1] = (y[60:120] - np.float32(60.0)) / np.float32(59.0)

    # Achromatic and chromatic value ramps retain the 0..2 scene scale.
    data[120:150, :, 1] = np.float32(0.0)
    data[120:150, :, 2] = np.float32(2.0) * horizontal[120:150]
    data[150:, :, 2] = np.float32(2.0) * horizontal[150:]
    return px.color.hsv_to_rgb(_hsv_frame(data))


def _photo_like_source() -> px.core.Frame:
    y, x = cp.indices((_HEIGHT, _WIDTH), dtype=cp.float32)
    normalized_x = x / np.float32(_WIDTH - 1)
    normalized_y = y / np.float32(_HEIGHT - 1)
    horizon = np.float32(0.58) + np.float32(0.04) * cp.sin(normalized_x * np.float32(15.0))
    sky = cp.clip((horizon - normalized_y) * np.float32(3.2), 0.0, 1.0)
    sun = cp.exp(
        -(
            (normalized_x - np.float32(0.70)) ** 2 / np.float32(0.003)
            + (normalized_y - np.float32(0.22)) ** 2 / np.float32(0.010)
        )
    )
    ridge = cp.where(
        normalized_y
        > np.float32(0.52)
        + np.float32(0.07) * cp.sin(normalized_x * np.float32(11.0))
        + np.float32(0.02) * cp.sin(normalized_x * np.float32(39.0)),
        np.float32(1.0),
        np.float32(0.0),
    )
    texture = cp.sin(x * np.float32(0.63) + y * np.float32(1.07)) * cp.sin(x * np.float32(1.73) - y * np.float32(0.31))
    data = cp.empty((_HEIGHT, _WIDTH, 3), dtype=cp.float32)
    data[..., 0] = np.float32(0.06) + np.float32(0.55) * sky + np.float32(1.75) * sun
    data[..., 1] = np.float32(0.08) + np.float32(0.80) * sky + np.float32(0.22) * ridge
    data[..., 2] = np.float32(0.10) + np.float32(1.10) * sky + np.float32(0.05) * ridge
    data[..., 0] += ridge * texture * np.float32(0.05)
    data[..., 1] += ridge * texture * np.float32(0.08)
    data[..., 2] += ridge * texture * np.float32(0.04)
    return _rgb_frame(cp.maximum(data, np.float32(0.0)))


def _color_wheel() -> px.core.Frame:
    y, x = cp.indices((_HEIGHT, _WIDTH), dtype=cp.float32)
    center_x = np.float32((_WIDTH - 1) / 2.0)
    center_y = np.float32((_HEIGHT - 1) / 2.0)
    normalized_x = (x - center_x) / np.float32(_HEIGHT / 2.0)
    normalized_y = (y - center_y) / np.float32(_HEIGHT / 2.0)
    radius = cp.hypot(normalized_x, normalized_y)
    data = cp.empty((_HEIGHT, _WIDTH, 3), dtype=cp.float32)
    data[..., 0] = cp.mod(cp.arctan2(normalized_y, normalized_x) / np.float32(2.0 * np.pi), np.float32(1.0))
    data[..., 1] = cp.clip(radius, np.float32(0.0), np.float32(1.0))
    data[..., 2] = cp.where(radius <= np.float32(1.0), np.float32(2.0), np.float32(0.0))
    return px.color.hsv_to_rgb(_hsv_frame(data))


def _display(frame: px.core.Frame) -> px.core.Frame:
    converted = px.color.rgb_to_rgb(frame, output_colorspace="sRGB", output_gamma="srgb")
    return px.io.from_array(
        cp.clip(converted.data, np.float32(0.0), np.float32(1.0)),
        colorspace="sRGB",
        gamma="srgb",
        channels="RGB",
    )


def _plane(hsv: px.core.Frame, label: str, *, scale: float = 1.0) -> px.core.Frame:
    index = hsv.channels.index(label)
    values = cp.clip(hsv.data[..., index] * np.float32(scale), np.float32(0.0), np.float32(1.0))
    rgb = cp.repeat(values[..., None], 3, axis=2)
    return px.io.from_array(rgb, colorspace="sRGB", gamma="srgb", channels="RGB")


def _error_view(source: px.core.Frame, restored: px.core.Frame) -> px.core.Frame:
    error = cp.clip(cp.abs(restored.data - source.data) * _ERROR_GAIN, np.float32(0.0), np.float32(1.0))
    return px.io.from_array(error, colorspace="sRGB", gamma="srgb", channels="RGB")


def _panel(frame: px.core.Frame, label: str) -> px.core.Frame:
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
        text=f"{label}\nraw {minimum:.3g}..{maximum:.3g}",
        position=(5.0, 12.0),
        size=8.0,
        color=(1.0, 1.0, 1.0),
        anchor="baseline-left",
        font="mono",
        line_spacing=0.9,
    )
    return px.transform.stack((bar, frame), direction="vertical")


def generate_sheet(path: Path) -> None:
    diagnostic = _diagnostic_chart()
    diagnostic_hsv = px.color.rgb_to_hsv(diagnostic)
    diagnostic_restored = px.color.hsv_to_rgb(diagnostic_hsv)
    photo = _photo_like_source()
    photo_hsv = px.color.rgb_to_hsv(photo)
    photo_restored = px.color.hsv_to_rgb(photo_hsv)

    diagnostic_row = px.transform.stack(
        (
            _panel(_display(diagnostic), "DIAGNOSTIC RGB / V up to 2"),
            _panel(_plane(diagnostic_hsv, "H"), "H PLANE / hue turn"),
            _panel(_plane(diagnostic_hsv, "S"), "S PLANE / 0..1"),
            _panel(_plane(diagnostic_hsv, "V", scale=0.5), "V PLANE / display divided by 2"),
            _panel(_display(_color_wheel()), "HSV WHEEL / six sectors + seam"),
        ),
        direction="horizontal",
    )
    round_trip_row = px.transform.stack(
        (
            _panel(_display(photo), "PHOTO-LIKE ORIGINAL"),
            _panel(_display(photo_restored), "PHOTO-LIKE RESTORED"),
            _panel(_error_view(photo, photo_restored), "PHOTO ERROR x262144"),
            _panel(_display(diagnostic_restored), "DIAGNOSTIC RESTORED"),
            _panel(_error_view(diagnostic, diagnostic_restored), "DIAGNOSTIC ERROR x262144"),
        ),
        direction="horizontal",
    )
    sheet = px.transform.stack((diagnostic_row, round_trip_row), direction="vertical")
    code = cp.rint(cp.clip(sheet.data, np.float32(0.0), np.float32(1.0)) * np.float32(255.0)).astype(cp.uint8)
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
