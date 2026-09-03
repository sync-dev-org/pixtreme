"""Generate deterministic visual comparisons for v1-red-tokens acceptance 74."""

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

_LOG_A = np.float64("0.224282")
_LOG_B = np.float64("155.975327")
_LOG_C = np.float64("0.01")
_LOG_G = np.float64("15.1927")
_CINEON_OFFSET = np.float64("0.0107977516232771")


def _piecewise(
    values: np.ndarray,
    mask: np.ndarray,
    lower: Callable[[np.ndarray], np.ndarray],
    upper: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    result[mask] = lower(values[mask])
    result[~mask] = upper(values[~mask])
    return result


def _log3g10_encode(values: np.ndarray) -> np.ndarray:
    shifted = values + _LOG_C
    return _piecewise(
        shifted,
        shifted < 0.0,
        lambda part: _LOG_G * part,
        lambda part: _LOG_A * np.log10(_LOG_B * part + 1.0),
    )


def _cineon_encode(values: np.ndarray) -> np.ndarray:
    sign = np.where(values < 0.0, -1.0, 1.0)
    magnitude = np.abs(values)
    encoded = (685.0 + 300.0 * np.log10(magnitude * (1.0 - _CINEON_OFFSET) + _CINEON_OFFSET)) / 1023.0
    return sign * encoded


def _public_curve(linear: np.ndarray, gamma: str) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.repeat(linear.astype(np.float32)[:, None], 3, axis=1)[None, :, :]
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
    draw.text((left, top + height + 4), f"x {x_range[0]:g} .. {x_range[1]:g}", fill=_TEXT, font=font)
    draw.text((left + width - 120, top + height + 4), f"y {y_range[0]:g} .. {y_range[1]:g}", fill=_TEXT, font=font)


def _transfer_sheet() -> Image.Image:
    image = Image.new("RGB", (_WIDTH, 1220), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    plot_width = _WIDTH - _LEFT - _RIGHT
    draw.text((_LEFT, 18), "RED transfer tokens: GPU float32 vs independent host float64", fill=_TEXT, font=font)
    draw.text((_LEFT, 38), "GPU", fill=_GPU, font=font)
    draw.text((_LEFT + 44, 38), "oracle", fill=_ORACLE, font=font)

    linear = np.linspace(-0.25, 2.0, _SAMPLES, dtype=np.float32)
    log_gpu, log_restored = _public_curve(linear, "RED-Log3G10")
    log_oracle = _log3g10_encode(linear.astype(np.float64))
    film_gpu, film_restored = _public_curve(linear, "REDlogFilm")
    film_oracle = _cineon_encode(linear.astype(np.float64))
    cineon_gpu, _ = _public_curve(linear, "Cineon")

    log_anchor_x = np.asarray((-0.01, 0.0, 0.18, 1.0), dtype=np.float64)
    log_anchor_gpu, _ = _public_curve(log_anchor_x, "RED-Log3G10")
    log_anchor_oracle = _log3g10_encode(log_anchor_x)
    film_anchor_x = np.asarray((0.0, 0.18, 1.0), dtype=np.float64)
    film_anchor_gpu, _ = _public_curve(film_anchor_x, "REDlogFilm")
    film_anchor_oracle = _cineon_encode(film_anchor_x)

    panels = (
        (
            "RED-Log3G10 signed encode",
            linear,
            log_gpu,
            log_oracle,
            log_anchor_x,
            log_anchor_gpu,
            log_anchor_oracle,
            (-0.25, 2.0),
            (-3.7, 0.57),
        ),
        (
            "REDlogFilm signed mirror",
            linear,
            film_gpu,
            film_oracle,
            film_anchor_x,
            film_anchor_gpu,
            film_anchor_oracle,
            (-0.25, 2.0),
            (-0.51, 0.76),
        ),
    )
    for index, (title, x, gpu, oracle, anchor_x, anchor_gpu, anchor_oracle, x_range, y_range) in enumerate(panels):
        box = (_LEFT, 80 + index * 250, plot_width, 190)
        _axes(draw, font, box=box, title=title, x_range=x_range, y_range=y_range)
        _plot(draw, x, oracle, box=box, x_range=x_range, y_range=y_range, color=_ORACLE)
        _plot(draw, x, gpu, box=box, x_range=x_range, y_range=y_range, color=_GPU)
        _markers(
            draw,
            anchor_x,
            anchor_oracle,
            box=box,
            x_range=x_range,
            y_range=y_range,
            color=_ORACLE,
            radius=5,
        )
        _markers(
            draw,
            anchor_x,
            anchor_gpu,
            box=box,
            x_range=x_range,
            y_range=y_range,
            color=_GPU,
            radius=2,
        )

    anchor_box = (_LEFT, 555, plot_width, 155)
    draw.rectangle(
        (anchor_box[0], anchor_box[1], anchor_box[0] + anchor_box[2], anchor_box[1] + anchor_box[3]),
        outline=_GRID,
    )
    draw.text(
        (anchor_box[0] + 8, anchor_box[1] + 8),
        "Published anchors (circle markers above): GPU float32 / independent host float64",
        fill=_TEXT,
        font=font,
    )
    anchor_rows = (
        *(
            ("RED-Log3G10", x, gpu, oracle)
            for x, gpu, oracle in zip(log_anchor_x, log_anchor_gpu, log_anchor_oracle, strict=True)
        ),
        *(
            ("REDlogFilm", x, gpu, oracle)
            for x, gpu, oracle in zip(film_anchor_x, film_anchor_gpu, film_anchor_oracle, strict=True)
        ),
    )
    for index, (name, x, gpu, oracle) in enumerate(anchor_rows):
        draw.text(
            (anchor_box[0] + 8, anchor_box[1] + 28 + index * 17),
            f"{name:<14} x={float(x):>9.6f}  GPU={float(gpu):>12.9f}  oracle={float(oracle):>12.9f}",
            fill=_TEXT,
            font=font,
        )

    cut_linear = np.linspace(-0.0102, -0.0098, _SAMPLES, dtype=np.float32)
    cut_gpu, _ = _public_curve(cut_linear, "RED-Log3G10")
    cut_oracle = _log3g10_encode(cut_linear.astype(np.float64))
    cut_box = (_LEFT, 760, plot_width // 2 - 12, 180)
    _axes(
        draw,
        font,
        box=cut_box,
        title="Log3G10 branch boundary at x=-0.01",
        x_range=(-0.0102, -0.0098),
        y_range=(-0.0031, 0.0031),
    )
    _plot(
        draw, cut_linear, cut_oracle, box=cut_box, x_range=(-0.0102, -0.0098), y_range=(-0.0031, 0.0031), color=_ORACLE
    )
    _plot(draw, cut_linear, cut_gpu, box=cut_box, x_range=(-0.0102, -0.0098), y_range=(-0.0031, 0.0031), color=_GPU)

    zero_linear = np.linspace(-0.001, 0.001, _SAMPLES, dtype=np.float32)
    zero_gpu, _ = _public_curve(zero_linear, "REDlogFilm")
    zero_oracle = _cineon_encode(zero_linear.astype(np.float64))
    zero_box = (_LEFT + plot_width // 2 + 12, 760, plot_width // 2 - 12, 180)
    _axes(
        draw,
        font,
        box=zero_box,
        title="REDlogFilm zero mirror",
        x_range=(-0.001, 0.001),
        y_range=(-0.11, 0.11),
    )
    _plot(draw, zero_linear, zero_oracle, box=zero_box, x_range=(-0.001, 0.001), y_range=(-0.11, 0.11), color=_ORACLE)
    _plot(draw, zero_linear, zero_gpu, box=zero_box, x_range=(-0.001, 0.001), y_range=(-0.11, 0.11), color=_GPU)

    residual_box = (_LEFT, 1000, plot_width, 150)
    residual = np.maximum(np.abs(log_restored - linear), np.abs(film_restored - linear))
    residual_upper = max(float(residual.max()) * 1.1, 1e-7)
    _axes(
        draw,
        font,
        box=residual_box,
        title="maximum absolute round-trip residual",
        x_range=(-0.25, 2.0),
        y_range=(0.0, residual_upper),
    )
    _plot(draw, linear, residual, box=residual_box, x_range=(-0.25, 2.0), y_range=(0.0, residual_upper), color=_ERROR)
    bit_difference_count = int(np.count_nonzero(film_gpu.view(np.uint64) != cineon_gpu.view(np.uint64)))
    draw.text(
        (_LEFT, 1190),
        f"REDlogFilm / Cineon float64-container bit differences after identical float32 GPU output: {bit_difference_count}",
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


def _gamut_strip(colorspace: str, width: int, height: int) -> np.ndarray:
    source_rgb = _source_colors(width)
    source = px.io.from_array(cp.asarray(source_rgb[None, :, :]), colorspace=colorspace, gamma="linear", channels="RGB")
    converted = px.color.rgb_to_rgb(source, output_colorspace="Rec.709", output_gamma="linear")
    linear = px.io.to_array(converted).get()[0]
    display = np.power(np.clip(linear, 0.0, 1.0), np.float32(1.0 / 2.2))
    row = np.rint(display * np.float32(255.0)).astype(np.uint8)
    return np.repeat(row[None, :, :], height, axis=0)


def _gamut_sheet() -> Image.Image:
    names = ("REDWideGamutRGB", "DRAGONcolor", "DRAGONcolor2", "REDcolor2", "REDcolor3", "REDcolor4")
    strip_width = _WIDTH - _LEFT - _RIGHT
    strip_height = 100
    image = Image.new("RGB", (_WIDTH, 780), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((_LEFT, 18), "Six independent RED gamuts converted to Rec.709 / linear", fill=_TEXT, font=font)
    draw.text(
        (_LEFT, 38),
        "Input ramp includes negative values and scene overshoot; display preview clips only after conversion",
        fill=_TEXT,
        font=font,
    )
    for index, colorspace in enumerate(names):
        top = 78 + index * 112
        image.paste(Image.fromarray(_gamut_strip(colorspace, strip_width, strip_height), mode="RGB"), (_LEFT, top))
        draw.text((12, top + 42), colorspace, fill=_TEXT, font=font)
        draw.rectangle((_LEFT, top, _LEFT + strip_width - 1, top + strip_height - 1), outline=_GRID)
    return image


def generate(directory: Path) -> tuple[Path, Path]:
    """Generate the deterministic transfer and gamut comparison images."""
    directory.mkdir(parents=True, exist_ok=True)
    transfer_path = directory / "red-transfer-curves.png"
    gamut_path = directory / "red-gamut-conversions.png"
    _transfer_sheet().save(transfer_path, format="PNG", compress_level=9)
    _gamut_sheet().save(gamut_path, format="PNG", compress_level=9)
    return transfer_path, gamut_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(".nf/tmp/sheets-17-red"))
    arguments = parser.parse_args()
    for path in generate(arguments.output_dir):
        print(path)


if __name__ == "__main__":
    main()
