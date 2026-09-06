"""Generate deterministic visual evidence for v1-canon-tokens acceptance 97."""

from __future__ import annotations

import argparse
from collections.abc import Callable
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
_ERROR = (243, 101, 128)
_SAMPLES = 2048

_CURVES = {
    "Canon-Log": {
        "a": np.float64("0.45310179"),
        "b": np.float64("10.1596"),
        "c": np.float64("0.12512248"),
    },
    "Canon-Log-2": {
        "a": np.float64("0.24136077"),
        "b": np.float64("87.099375"),
        "c": np.float64("0.092864125"),
    },
    "Canon-Log-3": {
        "a": np.float64("0.36726845"),
        "b": np.float64("14.98325"),
        "m": np.float64("1.9754798"),
        "c": np.float64("0.12512219"),
        "c_pos": np.float64("0.12240537"),
        "c_neg": np.float64("0.12783901"),
        "cut": np.float64("0.014"),
    },
}


def _piecewise(
    values: np.ndarray,
    masks_and_functions: tuple[tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]], ...],
) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    for mask, function in masks_and_functions:
        result[mask] = function(values[mask])
    return result


def _canon_encode(gamma: str, reflectance: np.ndarray) -> np.ndarray:
    constants = _CURVES[gamma]
    x = np.asarray(reflectance, dtype=np.float64) / np.float64("0.9")
    a, b, c = constants["a"], constants["b"], constants["c"]
    if gamma != "Canon-Log-3":
        positive = x >= 0.0
        return _piecewise(
            x,
            (
                (~positive, lambda part: -a * np.log10(1.0 - b * part) + c),
                (positive, lambda part: a * np.log10(1.0 + b * part) + c),
            ),
        )
    cut = constants["cut"]
    lower = x < -cut
    upper = x > cut
    return _piecewise(
        x,
        (
            (lower, lambda part: -a * np.log10(1.0 - b * part) + constants["c_neg"]),
            (~(lower | upper), lambda part: constants["m"] * part + c),
            (upper, lambda part: a * np.log10(1.0 + b * part) + constants["c_pos"]),
        ),
    )


def _public_curve(reflectance: np.ndarray, gamma: str) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.repeat(reflectance.astype(np.float32)[:, None], 3, axis=1)[None, :, :]
    source = px.io.from_array(cp.asarray(rgb), colorspace="ACEScg", gamma="linear", channels="RGB")
    encoded = px.color.linear_to_gamma(source, gamma=gamma)
    restored = px.color.gamma_to_linear(encoded, gamma=gamma)
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
    image = Image.new("RGB", (_WIDTH, 1600), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    plot_width = _WIDTH - _LEFT - _RIGHT
    draw.text((_LEFT, 18), "Canon Log transfer tokens: GPU float32 vs independent host float64", fill=_TEXT, font=font)
    draw.text((_LEFT, 38), "GPU", fill=_GPU, font=font)
    draw.text((_LEFT + 44, 38), "oracle", fill=_ORACLE, font=font)

    reflectance = np.linspace(-0.25, 1.5, _SAMPLES, dtype=np.float32)
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for gamma in _CURVES:
        gpu, restored = _public_curve(reflectance, gamma)
        curves[gamma] = (gpu, _canon_encode(gamma, reflectance.astype(np.float64)), restored)

    anchor_r = np.asarray((0.0, 0.18, 0.9), dtype=np.float64)
    for index, gamma in enumerate(_CURVES):
        gpu, oracle, _ = curves[gamma]
        anchor_gpu, _ = _public_curve(anchor_r, gamma)
        anchor_oracle = _canon_encode(gamma, anchor_r)
        box = (_LEFT, 82 + index * 250, plot_width, 190)
        _axes(
            draw,
            font,
            box=box,
            title=f"{gamma} signed encode (x = r / 0.9)",
            x_range=(-0.25, 1.5),
            y_range=(-0.35, 0.75),
        )
        _plot(draw, reflectance, oracle, box=box, x_range=(-0.25, 1.5), y_range=(-0.35, 0.75), color=_ORACLE)
        _plot(draw, reflectance, gpu, box=box, x_range=(-0.25, 1.5), y_range=(-0.35, 0.75), color=_GPU)
        _markers(
            draw,
            anchor_r,
            anchor_oracle,
            box=box,
            x_range=(-0.25, 1.5),
            y_range=(-0.35, 0.75),
            color=_ORACLE,
            radius=5,
        )
        _markers(
            draw,
            anchor_r,
            anchor_gpu,
            box=box,
            x_range=(-0.25, 1.5),
            y_range=(-0.35, 0.75),
            color=_GPU,
            radius=2,
        )
        draw.text(
            (_LEFT + 8, 82 + index * 250 + 8),
            " | ".join(
                f"r={float(r):.2f}: {float(gpu_value):.9f}/{float(oracle_value):.9f}"
                for r, gpu_value, oracle_value in zip(anchor_r, anchor_gpu, anchor_oracle, strict=True)
            ),
            fill=_TEXT,
            font=font,
        )

    cut = np.float32(_CURVES["Canon-Log-3"]["cut"] * np.float64("0.9"))
    cut_width = np.float32("0.00008")
    for index, (title, center) in enumerate((("negative", -cut), ("positive", cut))):
        cut_r = np.linspace(center - cut_width, center + cut_width, _SAMPLES, dtype=np.float32)
        cut_gpu, _ = _public_curve(cut_r, "Canon-Log-3")
        cut_oracle = _canon_encode("Canon-Log-3", cut_r.astype(np.float64))
        y_mid = float(_canon_encode("Canon-Log-3", np.asarray((center,), dtype=np.float64))[0])
        box = (_LEFT + index * (plot_width // 2 + 12), 848, plot_width // 2 - 12, 190)
        ranges = ((float(center - cut_width), float(center + cut_width)), (y_mid - 0.00022, y_mid + 0.00022))
        _axes(
            draw,
            font,
            box=box,
            title=f"Canon-Log-3 {title} cut: x={float(center / np.float32(0.9)):.3f}, r={float(center):.4f}",
            x_range=ranges[0],
            y_range=ranges[1],
        )
        _plot(draw, cut_r, cut_oracle, box=box, x_range=ranges[0], y_range=ranges[1], color=_ORACLE)
        _plot(draw, cut_r, cut_gpu, box=box, x_range=ranges[0], y_range=ranges[1], color=_GPU)

    residual = np.maximum.reduce(
        [np.abs(restored - reflectance.astype(np.float64)) for _, _, restored in curves.values()]
    )
    residual_upper = max(float(residual.max()) * 1.1, 1e-7)
    residual_box = (_LEFT, 1102, plot_width, 190)
    _axes(
        draw,
        font,
        box=residual_box,
        title="maximum absolute encode/decode round-trip residual across all three curves",
        x_range=(-0.25, 1.5),
        y_range=(0.0, residual_upper),
    )
    _plot(
        draw,
        reflectance,
        residual,
        box=residual_box,
        x_range=(-0.25, 1.5),
        y_range=(0.0, residual_upper),
        color=_ERROR,
    )

    draw.text(
        (_LEFT, 1355),
        "Published full-range code anchors: GPU float32 / independent host float64",
        fill=_TEXT,
        font=font,
    )
    anchor_sets = {
        "Canon-Log": np.asarray((0.0, 0.18, 0.9, 7.2), dtype=np.float64),
        "Canon-Log-2": np.asarray((0.0, 0.18, 0.9, 57.6), dtype=np.float64),
        "Canon-Log-3": np.asarray((0.0, 0.18, 0.9, 14.4), dtype=np.float64),
    }
    row = 0
    for gamma, values in anchor_sets.items():
        gpu, _ = _public_curve(values, gamma)
        oracle = _canon_encode(gamma, values)
        for r, gpu_value, oracle_value in zip(values, gpu, oracle, strict=True):
            draw.text(
                (_LEFT, 1375 + row * 17),
                f"{gamma:<12} r={float(r):>7.2f}  code={float(gpu_value * 1023.0):>13.8f} / "
                f"{float(oracle_value * 1023.0):>13.8f}",
                fill=_TEXT,
                font=font,
            )
            row += 1
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
    source = px.io.from_array(
        cp.asarray(source_rgb[None, :, :]), colorspace="Canon-Cinema-Gamut", gamma="linear", channels="RGB"
    )
    converted = px.color.rgb_to_rgb(source, output_colorspace="Rec.709", output_gamma="linear")
    rec709 = px.io.to_array(converted).get()[0]

    draw.text((_LEFT, 18), "Canon Cinema Gamut to Rec.709 / linear", fill=_TEXT, font=font)
    draw.text(
        (_LEFT, 38),
        "Scene-linear source includes negative values and overshoot; previews clip only for display",
        fill=_TEXT,
        font=font,
    )
    for top, label, values in (
        (86, "Canon source", source_rgb),
        (320, "Rec.709", rec709),
    ):
        image.paste(Image.fromarray(_display_strip(values, strip_height), mode="RGB"), (_LEFT, top))
        draw.text((38, top + 72), label, fill=_TEXT, font=font)
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
    transfer_path = directory / "canon-transfer-curves.png"
    gamut_path = directory / "canon-gamut-conversion.png"
    _transfer_sheet().save(transfer_path, format="PNG", compress_level=9)
    _gamut_sheet().save(gamut_path, format="PNG", compress_level=9)
    return transfer_path, gamut_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(".nf/tmp/sheets-17-canon"))
    arguments = parser.parse_args()
    for path in generate(arguments.output_dir):
        print(path)


if __name__ == "__main__":
    main()
