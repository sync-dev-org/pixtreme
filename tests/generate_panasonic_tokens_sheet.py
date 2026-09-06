"""Generate deterministic visual evidence for v1-panasonic-tokens acceptance 115."""

from __future__ import annotations

import argparse
from pathlib import Path

import cupy as cp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import pixtreme as px

_WIDTH = 1200
_LEFT = 150
_RIGHT = 36
_BACKGROUND = (17, 20, 27)
_GRID = (55, 62, 76)
_TEXT = (225, 229, 238)
_GPU = (72, 207, 205)
_ORACLE = (247, 197, 72)
_PRINTED = (175, 126, 242)
_ERROR = (243, 101, 128)
_SAMPLES = 4096

_A = np.float64("0.241514")
_B = np.float64("0.00873")
_C = np.float64("0.598206")
_LINEAR_CUT = np.float64("0.01")
_M = _A / ((_LINEAR_CUT + _B) * np.log(np.float64(10.0)))
_D = _A * np.log10(_LINEAR_CUT + _B) + _C - _M * _LINEAR_CUT


def _vlog_encode(reflectance: np.ndarray) -> np.ndarray:
    values = np.asarray(reflectance, dtype=np.float64)
    result = np.empty_like(values)
    linear = values < _LINEAR_CUT
    result[linear] = _M * values[linear] + _D
    result[~linear] = _A * np.log10(values[~linear] + _B) + _C
    return result


def _public_curve(reflectance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.repeat(reflectance.astype(np.float32)[:, None], 3, axis=1)[None, :, :]
    source = px.io.from_array(cp.asarray(rgb), colorspace="V-Gamut", gamma="linear", channels="RGB")
    encoded = px.color.linear_to_gamma(source, gamma="V-Log")
    restored = px.color.gamma_to_linear(encoded, gamma="V-Log")
    return (
        px.io.to_array(encoded).get()[0, :, 0].astype(np.float64),
        px.io.to_array(restored).get()[0, :, 0].astype(np.float64),
    )


def _map_x(values: np.ndarray, *, left: int, width: int, lower: float, upper: float) -> np.ndarray:
    return left + np.rint((values - lower) / (upper - lower) * width).astype(np.int32)


def _map_y(values: np.ndarray, *, top: int, height: int, lower: float, upper: float) -> np.ndarray:
    return top + height - np.rint((values - lower) / (upper - lower) * height).astype(np.int32)


def _plot(
    draw: ImageDraw.ImageDraw,
    x: np.ndarray,
    y: np.ndarray,
    *,
    box: tuple[int, int, int, int],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    color: tuple[int, int, int],
) -> None:
    left, top, width, height = box
    points = tuple(
        zip(
            _map_x(x, left=left, width=width, lower=x_range[0], upper=x_range[1]),
            _map_y(y, top=top, height=height, lower=y_range[0], upper=y_range[1]),
            strict=True,
        )
    )
    draw.line(points, fill=color, width=2)


def _markers(
    draw: ImageDraw.ImageDraw,
    x: np.ndarray,
    y: np.ndarray,
    *,
    box: tuple[int, int, int, int],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    color: tuple[int, int, int],
    radius: int,
) -> None:
    left, top, width, height = box
    x_pixels = _map_x(x, left=left, width=width, lower=x_range[0], upper=x_range[1])
    y_pixels = _map_y(y, top=top, height=height, lower=y_range[0], upper=y_range[1])
    for x_pixel, y_pixel in zip(x_pixels, y_pixels, strict=True):
        draw.ellipse(
            (x_pixel - radius, y_pixel - radius, x_pixel + radius, y_pixel + radius),
            outline=color,
            width=2,
        )


def _axes(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    *,
    box: tuple[int, int, int, int],
    title: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> None:
    left, top, width, height = box
    draw.rectangle((left, top, left + width, top + height), outline=_GRID)
    if x_range[0] <= 0.0 <= x_range[1]:
        x_zero = int(_map_x(np.asarray((0.0,)), left=left, width=width, lower=x_range[0], upper=x_range[1])[0])
        draw.line((x_zero, top, x_zero, top + height), fill=_GRID)
    if y_range[0] <= 0.0 <= y_range[1]:
        y_zero = int(_map_y(np.asarray((0.0,)), top=top, height=height, lower=y_range[0], upper=y_range[1])[0])
        draw.line((left, y_zero, left + width, y_zero), fill=_GRID)
    draw.text((left, top - 22), title, fill=_TEXT, font=font)
    draw.text((left, top + height + 4), f"r {x_range[0]:g} .. {x_range[1]:g}", fill=_TEXT, font=font)
    draw.text((left + width - 120, top + height + 4), f"y {y_range[0]:g} .. {y_range[1]:g}", fill=_TEXT, font=font)


def _transfer_sheet() -> Image.Image:
    image = Image.new("RGB", (_WIDTH, 1510), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    plot_width = _WIDTH - _LEFT - _RIGHT
    draw.text((_LEFT, 18), "Panasonic V-Log: GPU float32 vs independent host float64", fill=_TEXT, font=font)
    draw.text((_LEFT, 38), "GPU", fill=_GPU, font=font)
    draw.text((_LEFT + 44, 38), "tangent oracle", fill=_ORACLE, font=font)
    draw.text((_LEFT + 150, 38), "printed 5.6 / 0.125", fill=_PRINTED, font=font)

    signed = np.linspace(-0.25, 1.5, _SAMPLES, dtype=np.float64).astype(np.float32)
    signed_gpu, signed_restored = _public_curve(signed)
    signed_oracle = _vlog_encode(signed.astype(np.float64))
    signed_box = (_LEFT, 82, plot_width, 220)
    _axes(
        draw,
        font,
        box=signed_box,
        title="signed reflectance ramp with scene overshoot",
        x_range=(-0.25, 1.5),
        y_range=(-1.28, 0.73),
    )
    _plot(draw, signed, signed_oracle, box=signed_box, x_range=(-0.25, 1.5), y_range=(-1.28, 0.73), color=_ORACLE)
    _plot(draw, signed, signed_gpu, box=signed_box, x_range=(-0.25, 1.5), y_range=(-1.28, 0.73), color=_GPU)

    full = np.linspace(-0.5, 64.0, _SAMPLES, dtype=np.float64).astype(np.float32)
    full_gpu, _ = _public_curve(full)
    full_oracle = _vlog_encode(full.astype(np.float64))
    full_box = (_LEFT, 366, plot_width, 220)
    _axes(
        draw,
        font,
        box=full_box,
        title="full numeric contract: unbounded linear lower branch and logarithmic upper branch",
        x_range=(-0.5, 64.0),
        y_range=(-2.7, 1.1),
    )
    _plot(draw, full, full_oracle, box=full_box, x_range=(-0.5, 64.0), y_range=(-2.7, 1.1), color=_ORACLE)
    _plot(draw, full, full_gpu, box=full_box, x_range=(-0.5, 64.0), y_range=(-2.7, 1.1), color=_GPU)

    cut_width = np.float32("0.000004")
    cut = np.float32(_LINEAR_CUT)
    cut_r = np.linspace(cut - cut_width, cut + cut_width, _SAMPLES, dtype=np.float64).astype(np.float32)
    cut_gpu, _ = _public_curve(cut_r)
    cut_oracle = _vlog_encode(cut_r.astype(np.float64))
    printed = np.where(
        cut_r.astype(np.float64) < _LINEAR_CUT,
        np.float64("5.6") * cut_r.astype(np.float64) + np.float64("0.125"),
        _A * np.log10(cut_r.astype(np.float64) + _B) + _C,
    )
    cut_mid = float(_vlog_encode(np.asarray((cut,), dtype=np.float64))[0])
    cut_box = (_LEFT, 650, plot_width, 220)
    cut_y = (cut_mid - 0.000025, cut_mid + 0.000025)
    cut_x = (float(cut - cut_width), float(cut + cut_width))
    _axes(
        draw,
        font,
        box=cut_box,
        title="cut zoom: tangent continuity vs printed lower branch",
        x_range=cut_x,
        y_range=cut_y,
    )
    _plot(draw, cut_r, printed, box=cut_box, x_range=cut_x, y_range=cut_y, color=_PRINTED)
    _plot(draw, cut_r, cut_oracle, box=cut_box, x_range=cut_x, y_range=cut_y, color=_ORACLE)
    _plot(draw, cut_r, cut_gpu, box=cut_box, x_range=cut_x, y_range=cut_y, color=_GPU)

    residual = np.abs(signed_restored - signed.astype(np.float64))
    residual_upper = max(float(residual.max()) * 1.1, 1e-7)
    residual_box = (_LEFT, 934, plot_width, 200)
    _axes(
        draw,
        font,
        box=residual_box,
        title="absolute encode/decode round-trip residual",
        x_range=(-0.25, 1.5),
        y_range=(0.0, residual_upper),
    )
    _plot(
        draw,
        signed,
        residual,
        box=residual_box,
        x_range=(-0.25, 1.5),
        y_range=(0.0, residual_upper),
        color=_ERROR,
    )

    anchors = np.asarray((0.0, 0.18, 0.9, 1.0, 64.0), dtype=np.float32)
    anchor_gpu, _ = _public_curve(anchors)
    anchor_oracle = _vlog_encode(anchors.astype(np.float64))
    _markers(
        draw,
        anchors[:4],
        anchor_gpu[:4],
        box=signed_box,
        x_range=(-0.25, 1.5),
        y_range=(-1.28, 0.73),
        color=_GPU,
        radius=3,
    )
    draw.text(
        (_LEFT, 1190),
        "Numeric panel: reflectance, GPU encoded, oracle encoded, full-range 10-bit code",
        fill=_TEXT,
        font=font,
    )
    for row, (r, gpu_value, oracle_value) in enumerate(zip(anchors, anchor_gpu, anchor_oracle, strict=True)):
        draw.text(
            (_LEFT, 1212 + row * 22),
            f"r={float(r):>8.3f}  y={float(gpu_value):>13.9f} / {float(oracle_value):>13.9f}  "
            f"code={float(gpu_value * 1023.0):>13.8f}",
            fill=_TEXT,
            font=font,
        )
    draw.text(
        (_LEFT, 1340),
        f"max |GPU - oracle| signed={float(np.max(np.abs(signed_gpu - signed_oracle))):.3e}; "
        f"full={float(np.max(np.abs(full_gpu - full_oracle))):.3e}",
        fill=_TEXT,
        font=font,
    )
    draw.text(
        (_LEFT, 1362),
        f"max round-trip residual={float(residual.max()):.3e}; no display clipping is applied to numeric curves",
        fill=_TEXT,
        font=font,
    )
    return image


def _source_colors(width: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    return np.stack(
        (
            np.float32(1.35) * x - np.float32(0.15),
            np.float32(0.18) + np.float32(0.82) * np.sin(np.pi * x).astype(np.float32),
            np.float32(1.2) * (np.float32(1.0) - x) - np.float32(0.1),
        ),
        axis=1,
    )


def _display_strip(values: np.ndarray, height: int) -> np.ndarray:
    display = np.power(np.clip(values, 0.0, 1.0), np.float32(1.0 / 2.2))
    row = np.rint(display * np.float32(255.0)).astype(np.uint8)
    return np.repeat(row[None, :, :], height, axis=0)


def _gamut_sheet() -> Image.Image:
    strip_width = _WIDTH - _LEFT - _RIGHT
    strip_height = 170
    image = Image.new("RGB", (_WIDTH, 570), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    source_rgb = _source_colors(strip_width)
    source = px.io.from_array(cp.asarray(source_rgb[None, :, :]), colorspace="V-Gamut", gamma="linear", channels="RGB")
    converted = px.color.rgb_to_rgb(source, output_colorspace="Rec.709", output_gamma="linear")
    rec709 = px.io.to_array(converted).get()[0]

    draw.text((_LEFT, 18), "Panasonic V-Gamut to Rec.709 / linear", fill=_TEXT, font=font)
    draw.text(
        (_LEFT, 38),
        "Scene-linear source includes negative values and overshoot; previews clip only for display",
        fill=_TEXT,
        font=font,
    )
    for top, label, values in ((86, "V-Gamut source", source_rgb), (320, "Rec.709 conversion", rec709)):
        image.paste(Image.fromarray(_display_strip(values, strip_height), mode="RGB"), (_LEFT, top))
        draw.text((28, top + 72), label, fill=_TEXT, font=font)
        draw.rectangle((_LEFT, top, _LEFT + strip_width - 1, top + strip_height - 1), outline=_GRID)
        draw.text(
            (_LEFT, top + strip_height + 8),
            f"linear min={float(values.min()):.7f}, max={float(values.max()):.7f}",
            fill=_TEXT,
            font=font,
        )
    return image


def generate(directory: Path) -> tuple[Path, Path]:
    """Generate deterministic transfer and gamut comparison images."""
    directory.mkdir(parents=True, exist_ok=True)
    transfer_path = directory / "panasonic-vlog-transfer.png"
    gamut_path = directory / "panasonic-v-gamut-conversion.png"
    _transfer_sheet().save(transfer_path, format="PNG", compress_level=9)
    _gamut_sheet().save(gamut_path, format="PNG", compress_level=9)
    return transfer_path, gamut_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(".nf/tmp/sheets-17-panasonic"))
    arguments = parser.parse_args()
    for path in generate(arguments.output_dir):
        print(path)


if __name__ == "__main__":
    main()
