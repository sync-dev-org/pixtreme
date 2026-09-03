"""Generate the manual visual-acceptance sheet for v1-tonemap-bt2408 acceptance 16."""

from __future__ import annotations

import argparse
from pathlib import Path

import cupy as cp
import numpy as np

import pixtreme as px

_WIDTH = 320
_HEIGHT = 180
_LABEL_HEIGHT = 38


def _gain(output_gamma: str) -> np.float32:
    if output_gamma == "PQ":
        return np.float32(np.float64(203) / np.float64(10000))
    a = np.float64(0.17883277)
    b = np.float64(1) - np.float64(4) * a
    c = np.float64(0.5) - a * np.log(np.float64(4) * a)
    return np.float32((np.exp((np.float64(0.75) - c) / a) + b) / np.float64(12))


def _source() -> px.core.Frame:
    y, x = cp.indices((_HEIGHT, _WIDTH), dtype=cp.float32)
    horizontal = x / np.float32(_WIDTH - 1)
    vertical = y / np.float32(_HEIGHT - 1)
    data = cp.empty((_HEIGHT, _WIDTH, 3), dtype=cp.float32)

    # A smooth signed-to-superwhite SDR ramp reveals banding, clipping, and white placement.
    ramp = np.float32(-0.125) + np.float32(2.125) * horizontal
    data[..., 0] = ramp
    data[..., 1] = ramp * (np.float32(0.65) + np.float32(0.35) * vertical)
    data[..., 2] = ramp * (np.float32(1.0) - np.float32(0.45) * vertical)

    # Neutral, primary, and secondary patches use exact scene values, including reference white and superwhite.
    patches = cp.asarray(
        (
            (-0.125, -0.125, -0.125),
            (0.0, 0.0, 0.0),
            (0.18, 0.18, 0.18),
            (1.0, 1.0, 1.0),
            (1.5, 1.5, 1.5),
            (2.0, 2.0, 2.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 1.0),
            (1.0, 0.0, 1.0),
        ),
        dtype=cp.float32,
    )
    patch_width = _WIDTH // 6
    patch_height = 34
    for row in range(2):
        for column in range(6):
            left = column * patch_width
            right = _WIDTH if column == 5 else (column + 1) * patch_width
            top = row * patch_height
            data[top : top + patch_height, left:right] = patches[row * 6 + column]
    return px.io.from_array(data, colorspace="Rec.709", gamma="linear", channels="RGB")


def _raw_signal(frame: px.core.Frame) -> px.core.Frame:
    return px.io.from_array(
        cp.clip(frame.data, np.float32(0.0), np.float32(1.0)),
        colorspace="sRGB",
        gamma="sRGB",
        channels="RGB",
    )


def _white_normalized_linear(frame: px.core.Frame, *, scale: np.float32) -> px.core.Frame:
    linear = px.color.rgb_to_rgb(frame, output_gamma="linear")
    return px.io.from_array(
        linear.data * scale,
        colorspace="Rec.2020",
        gamma="linear",
        channels="RGB",
    )


def _preview(frame: px.core.Frame) -> px.core.Frame:
    display = px.color.rgb_to_rgb(frame, output_colorspace="sRGB", output_gamma="sRGB")
    return px.io.from_array(
        cp.clip(display.data, np.float32(0.0), np.float32(1.0)),
        colorspace="sRGB",
        gamma="sRGB",
        channels="RGB",
    )


def _range_diagnostic(frame: px.core.Frame) -> px.core.Frame:
    minimum = cp.min(frame.data, axis=2)
    maximum = cp.max(frame.data, axis=2)
    diagnostic = cp.empty_like(frame.data)
    diagnostic[..., 0] = cp.clip(-minimum * np.float32(4.0), np.float32(0.0), np.float32(1.0))
    diagnostic[..., 1] = cp.clip(cp.mean(frame.data, axis=2), np.float32(0.0), np.float32(1.0))
    diagnostic[..., 2] = cp.clip(maximum - np.float32(1.0), np.float32(0.0), np.float32(1.0))
    return px.io.from_array(diagnostic, colorspace="sRGB", gamma="sRGB", channels="RGB")


def _panel(frame: px.core.Frame, label: str, *, measured: px.core.Frame) -> px.core.Frame:
    minimum = float(cp.min(measured.data).get())
    maximum = float(cp.max(measured.data).get())
    bar = px.io.from_array(
        cp.full((_LABEL_HEIGHT, _WIDTH, 3), np.float32(0.015), dtype=cp.float32),
        colorspace="sRGB",
        gamma="sRGB",
        channels="RGB",
    )
    bar = px.draw.text(
        bar,
        text=f"{label}\nraw {minimum:.4f}..{maximum:.4f}",
        position=(5.0, 12.0),
        size=8.0,
        color=(1.0, 1.0, 1.0),
        anchor="baseline-left",
        font="mono",
        line_spacing=0.9,
    )
    return px.transform.stack((bar, frame), direction="vertical")


def generate_sheet(path: Path) -> None:
    source = _source()
    hlg_none = px.color.rgb_to_rgb(source, output_colorspace="Rec.2020", output_gamma="HLG")
    hlg_bt2408 = px.color.rgb_to_rgb(
        source,
        output_colorspace="Rec.2020",
        output_gamma="HLG",
        tonemap="BT.2408",
    )
    pq_none = px.color.rgb_to_rgb(source, output_colorspace="Rec.2020", output_gamma="PQ")
    pq_bt2408 = px.color.rgb_to_rgb(
        source,
        output_colorspace="Rec.2020",
        output_gamma="PQ",
        tonemap="BT.2408",
    )

    hlg_none_linear = _white_normalized_linear(hlg_none, scale=np.float32(1.0))
    hlg_bt2408_normalized = _white_normalized_linear(hlg_bt2408, scale=np.float32(1.0) / _gain("HLG"))
    pq_none_linear = _white_normalized_linear(pq_none, scale=np.float32(1.0))
    pq_bt2408_normalized = _white_normalized_linear(pq_bt2408, scale=np.float32(1.0) / _gain("PQ"))

    raw_row = px.transform.stack(
        (
            _panel(_raw_signal(hlg_none), "HLG NONE / raw signal", measured=hlg_none),
            _panel(_raw_signal(hlg_bt2408), "HLG BT.2408 / white = 0.75", measured=hlg_bt2408),
            _panel(_raw_signal(pq_none), "PQ NONE / raw signal", measured=pq_none),
            _panel(_raw_signal(pq_bt2408), "PQ BT.2408 / white = ST2084(203)", measured=pq_bt2408),
        ),
        direction="horizontal",
    )
    preview_row = px.transform.stack(
        (
            _panel(_preview(hlg_none_linear), "HLG NONE / SDR preview", measured=hlg_none_linear),
            _panel(
                _preview(hlg_bt2408_normalized),
                "HLG BT.2408 / divide by G_HLG",
                measured=hlg_bt2408_normalized,
            ),
            _panel(_preview(pq_none_linear), "PQ NONE / SDR preview", measured=pq_none_linear),
            _panel(
                _preview(pq_bt2408_normalized),
                "PQ BT.2408 / divide by 203/10000",
                measured=pq_bt2408_normalized,
            ),
        ),
        direction="horizontal",
    )
    diagnostic_row = px.transform.stack(
        (
            _panel(_range_diagnostic(hlg_none_linear), "HLG NONE / red<0 blue>1", measured=hlg_none_linear),
            _panel(
                _range_diagnostic(hlg_bt2408_normalized),
                "HLG BT.2408 normalized / red<0 blue>1",
                measured=hlg_bt2408_normalized,
            ),
            _panel(_range_diagnostic(pq_none_linear), "PQ NONE / red<0 blue>1", measured=pq_none_linear),
            _panel(
                _range_diagnostic(pq_bt2408_normalized),
                "PQ BT.2408 normalized / red<0 blue>1",
                measured=pq_bt2408_normalized,
            ),
        ),
        direction="horizontal",
    )
    sheet = px.transform.stack((raw_row, preview_row, diagnostic_row), direction="vertical")
    code = cp.rint(cp.clip(sheet.data, np.float32(0.0), np.float32(1.0)) * np.float32(255.0)).astype(cp.uint8)
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
