"""Generate deterministic visual comparisons for v1-sony-tokens acceptance 14."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import cupy as cp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import pixtreme as px

_WIDTH = 1200
_HEIGHT = 680
_LEFT = 96
_RIGHT = 30
_PLOT_WIDTH = _WIDTH - _LEFT - _RIGHT
_SAMPLES = 2048

_BACKGROUND = (17, 20, 27)
_GRID = (55, 62, 76)
_TEXT = (225, 229, 238)
_GPU = (72, 207, 205)
_ORACLE = (247, 197, 72)
_ERROR = (243, 101, 128)
_C = np.float64("0.030001222851889303")


def _piecewise(
    values: np.ndarray,
    cut: float,
    lower: Callable[[np.ndarray], np.ndarray],
    upper: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    lower_mask = values < cut
    result[lower_mask] = lower(values[lower_mask])
    result[~lower_mask] = upper(values[~lower_mask])
    return result


def _sony_encode(values: np.ndarray, *, slog2: bool) -> np.ndarray:
    x = values / np.float64(0.9)
    scale = np.float64(155.0) / np.float64(219.0) if slog2 else np.float64(1.0)
    slope = np.float64("3.53881278538813") if slog2 else np.float64(5.0)
    y = _piecewise(
        x,
        np.float64(0.0),
        lambda part: slope * part + _C,
        lambda part: (
            np.float64(0.432699) * np.log10(scale * part + np.float64(0.037584))
            + np.float64(0.616596)
            + np.float64(0.03)
        ),
    )
    return (np.float64(64.0) + np.float64(876.0) * y) / np.float64(1023.0)


def _public_curve(linear: np.ndarray, gamma: str) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.repeat(linear.astype(np.float32)[:, None], 3, axis=1)[None, :, :]
    source = px.io.from_array(cp.asarray(rgb), colorspace="ACEScg", gamma="linear", channels="RGB")
    encoded = px.color.linear_to_gamma(source, gamma=gamma)
    round_trip = px.color.gamma_to_linear(encoded, gamma=gamma)
    return (
        px.io.to_array(encoded).get()[0, :, 0].astype(np.float64),
        px.io.to_array(round_trip).get()[0, :, 0].astype(np.float64),
    )


def _x_pixel(values: np.ndarray, value: float) -> int:
    fraction = (value - float(values[0])) / (float(values[-1]) - float(values[0]))
    return _LEFT + int(round(fraction * (_PLOT_WIDTH - 1)))


def _draw_markers(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    values: np.ndarray,
    markers: Sequence[tuple[float, str]],
    top: int,
    bottom: int,
) -> None:
    for value, label in markers:
        x = _x_pixel(values, value)
        draw.line((x, top, x, bottom), fill=_GRID, width=1)
        bounds = draw.textbbox((0, 0), label, font=font)
        label_width = bounds[2] - bounds[0]
        label_x = min(x + 3, _WIDTH - _RIGHT - label_width)
        draw.text((label_x, bottom + 3), label, fill=_TEXT, font=font)


def _draw_curve(
    draw: ImageDraw.ImageDraw,
    values: np.ndarray,
    curve: np.ndarray,
    *,
    top: int,
    bottom: int,
    lower: float,
    upper: float,
    color: tuple[int, int, int],
) -> None:
    normalized = np.clip((curve - lower) / (upper - lower), np.float64(0.0), np.float64(1.0))
    x = np.linspace(_LEFT, _LEFT + _PLOT_WIDTH - 1, values.size)
    y = bottom - normalized * (bottom - top)
    draw.line(tuple(zip(x.tolist(), y.tolist(), strict=True)), fill=color, width=2)


def _draw_strip(image: Image.Image, curve: np.ndarray, *, top: int, lower: float, upper: float) -> None:
    normalized = np.clip((curve - lower) / (upper - lower), np.float64(0.0), np.float64(1.0))
    samples = np.rint(normalized * np.float64(255.0)).astype(np.uint8)
    rgb = np.repeat(np.repeat(samples[None, :, None], 3, axis=2), 32, axis=0)
    resized = Image.fromarray(rgb, mode="RGB").resize((_PLOT_WIDTH, 32), resample=Image.Resampling.NEAREST)
    image.paste(resized, (_LEFT, top))


def _curve_sheet(gamma: str, *, slog2: bool, anchors: tuple[int, int, int]) -> Image.Image:
    linear = np.linspace(-0.3, 1.5, _SAMPLES, dtype=np.float32).astype(np.float64)
    gpu_encoded, gpu_round_trip = _public_curve(linear, gamma)
    oracle_encoded = _sony_encode(linear, slog2=slog2)
    encoded_error = gpu_encoded - oracle_encoded
    round_trip_error = gpu_round_trip - linear
    np.testing.assert_allclose(gpu_encoded, oracle_encoded, rtol=0.0, atol=2.0e-6)
    np.testing.assert_allclose(gpu_round_trip, linear, rtol=0.0, atol=4.0e-6)

    image = Image.new("RGB", (_WIDTH, _HEIGHT), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((_LEFT, 18), f"{gamma}: signed piecewise encode and round-trip", fill=_TEXT, font=font)
    draw.text((_LEFT, 38), "GPU float32", fill=_GPU, font=font)
    draw.text((_LEFT + 92, 38), "independent host float64 oracle", fill=_ORACLE, font=font)
    draw.text(
        (_LEFT + 300, 38),
        f"rounded 10-bit anchors: {anchors[0]} / {anchors[1]} / {anchors[2]}",
        fill=_TEXT,
        font=font,
    )

    markers = ((-0.25, "negative"), (0.0, "branch/0%"), (0.18, "18%"), (0.9, "90%"), (1.5, "overshoot"))
    lower = float(min(gpu_encoded.min(), oracle_encoded.min()))
    upper = float(max(gpu_encoded.max(), oracle_encoded.max()))
    margin = (upper - lower) * 0.04
    lower -= margin
    upper += margin

    draw.text((12, 82), "input", fill=_TEXT, font=font)
    draw.text((12, 121), "encoded", fill=_GPU, font=font)
    _draw_strip(image, linear, top=72, lower=-0.3, upper=1.5)
    _draw_strip(image, gpu_encoded, top=111, lower=lower, upper=upper)
    _draw_markers(draw, font, linear, markers, 68, 143)

    encode_top, encode_bottom = 188, 430
    draw.rectangle((_LEFT, encode_top, _LEFT + _PLOT_WIDTH, encode_bottom), outline=_GRID)
    draw.text((12, encode_top), "encode", fill=_TEXT, font=font)
    _draw_curve(
        draw,
        linear,
        oracle_encoded,
        top=encode_top,
        bottom=encode_bottom,
        lower=lower,
        upper=upper,
        color=_ORACLE,
    )
    _draw_curve(
        draw,
        linear,
        gpu_encoded,
        top=encode_top,
        bottom=encode_bottom,
        lower=lower,
        upper=upper,
        color=_GPU,
    )
    _draw_markers(draw, font, linear, markers, encode_top, encode_bottom)

    encode_limit = max(abs(float(encoded_error.min())), abs(float(encoded_error.max())), 1.0e-7)
    encode_error_top, encode_error_bottom = 470, 535
    draw.rectangle((_LEFT, encode_error_top, _LEFT + _PLOT_WIDTH, encode_error_bottom), outline=_GRID)
    draw.text((12, encode_error_top), "GPU-oracle", fill=_ERROR, font=font)
    _draw_curve(
        draw,
        linear,
        encoded_error,
        top=encode_error_top,
        bottom=encode_error_bottom,
        lower=-encode_limit,
        upper=encode_limit,
        color=_ERROR,
    )
    _draw_markers(draw, font, linear, markers, encode_error_top, encode_error_bottom)

    round_trip_limit = max(abs(float(round_trip_error.min())), abs(float(round_trip_error.max())), 1.0e-7)
    round_trip_top, round_trip_bottom = 572, 637
    draw.rectangle((_LEFT, round_trip_top, _LEFT + _PLOT_WIDTH, round_trip_bottom), outline=_GRID)
    draw.text((12, round_trip_top), "round-trip", fill=_ERROR, font=font)
    _draw_curve(
        draw,
        linear,
        round_trip_error,
        top=round_trip_top,
        bottom=round_trip_bottom,
        lower=-round_trip_limit,
        upper=round_trip_limit,
        color=_ERROR,
    )
    _draw_markers(draw, font, linear, markers, round_trip_top, round_trip_bottom)
    return image


def _linear_rgb_strip(values: np.ndarray, *, lower: float, upper: float, height: int = 72) -> Image.Image:
    normalized = np.clip((values - lower) / (upper - lower), np.float64(0.0), np.float64(1.0))
    samples = np.rint(normalized * np.float64(255.0)).astype(np.uint8)
    return Image.fromarray(samples[None, :, :], mode="RGB").resize(
        (_PLOT_WIDTH, height), resample=Image.Resampling.NEAREST
    )


def _gamut_target(source_rgb: np.ndarray, colorspace: str) -> np.ndarray:
    frame = px.io.from_array(
        cp.asarray(source_rgb.astype(np.float32)[None, :, :]),
        colorspace=colorspace,
        gamma="linear",
        channels="RGB",
    )
    target = px.color.rgb_to_rgb(frame, output_colorspace="Rec.709", output_gamma="linear")
    return px.io.to_array(target).get()[0]


def _gamut_sheet() -> Image.Image:
    x = np.linspace(-0.25, 1.5, _SAMPLES, dtype=np.float32)
    source_rgb = np.stack(
        (
            x,
            np.float32(0.45) + np.float32(0.65) * np.sin(np.float32(4.0) * x),
            np.float32(1.25) - np.float32(0.75) * x,
        ),
        axis=1,
    )
    sgamut = _gamut_target(source_rgb, "S-Gamut")
    sgamut3 = _gamut_target(source_rgb, "S-Gamut3")
    bit_equal = np.array_equal(sgamut.view(np.uint32), sgamut3.view(np.uint32))
    if not bit_equal:
        raise AssertionError("S-Gamut and S-Gamut3 target transforms are not float32 bit-identical")

    image = Image.new("RGB", (_WIDTH, 470), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((_LEFT, 18), "S-Gamut / S-Gamut3 -> Rec.709 linear equivalence", fill=_TEXT, font=font)
    draw.text(
        (_LEFT, 38),
        "GPU float32: bit-identical=True | max abs delta=0 | display strips share one fixed linear scale",
        fill=_GPU,
        font=font,
    )
    lower = float(min(source_rgb.min(), sgamut.min(), sgamut3.min()))
    upper = float(max(source_rgb.max(), sgamut.max(), sgamut3.max()))
    rows = (
        ("source", source_rgb),
        ("S-Gamut", sgamut),
        ("S-Gamut3", sgamut3),
    )
    for index, (label, values) in enumerate(rows):
        top = 82 + index * 106
        draw.text((12, top + 28), label, fill=_TEXT, font=font)
        image.paste(_linear_rgb_strip(values, lower=lower, upper=upper), (_LEFT, top))
    difference = np.abs(sgamut.astype(np.float64) - sgamut3.astype(np.float64))
    difference_image = _linear_rgb_strip(difference, lower=0.0, upper=1.0e-7, height=42)
    draw.text((12, 405), "abs delta x1e7", fill=_ERROR, font=font)
    image.paste(difference_image, (_LEFT, 394))
    return image


def generate_sheets(directory: Path) -> tuple[Path, Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    slog_path = directory / "slog-signed-curve.png"
    slog2_path = directory / "slog2-signed-curve.png"
    gamut_path = directory / "sgamut-equivalence.png"
    _curve_sheet("S-Log", slog2=False, anchors=(90, 394, 636)).save(slog_path, format="PNG", compress_level=9)
    _curve_sheet("S-Log2", slog2=True, anchors=(90, 347, 582)).save(slog2_path, format="PNG", compress_level=9)
    _gamut_sheet().save(gamut_path, format="PNG", compress_level=9)
    return slog_path, slog2_path, gamut_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    arguments = parser.parse_args()
    generate_sheets(arguments.output_directory)


if __name__ == "__main__":
    main()
