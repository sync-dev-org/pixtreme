"""Generate deterministic visual comparisons for the S-Log3 / ARRI-LogC4 negative-extension correction."""

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
_LEFT = 92
_RIGHT = 30
_PLOT_WIDTH = _WIDTH - _LEFT - _RIGHT
_SAMPLES = 2048

_BACKGROUND = (17, 20, 27)
_GRID = (55, 62, 76)
_TEXT = (225, 229, 238)
_OLD = (243, 101, 128)
_TARGET = (72, 207, 205)
_DIFFERENCE = (247, 197, 72)

_SLOG3_CODE_CUT = np.float64(171.2102946929) / np.float64(1023.0)
_LOGC4_A = (np.float64(2.0) ** np.float64(18.0) - np.float64(16.0)) / np.float64(117.45)
_LOGC4_B = (np.float64(1023.0) - np.float64(95.0)) / np.float64(1023.0)
_LOGC4_C = np.float64(95.0) / np.float64(1023.0)
_LOGC4_S = (
    np.float64(7.0)
    * np.log(np.float64(2.0))
    * np.float64(2.0) ** (np.float64(7.0) - np.float64(14.0) * _LOGC4_C / _LOGC4_B)
) / (_LOGC4_A * _LOGC4_B)
_LOGC4_T = (
    np.float64(2.0) ** (np.float64(14.0) * (-_LOGC4_C / _LOGC4_B) + np.float64(6.0)) - np.float64(64.0)
) / _LOGC4_A


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


def _slog3_encode(values: np.ndarray) -> np.ndarray:
    return _piecewise(
        values,
        np.float64(0.01125),
        lambda x: (
            (x * (np.float64(171.2102946929) - np.float64(95.0)) / np.float64(0.01125) + np.float64(95.0))
            / np.float64(1023.0)
        ),
        lambda x: (
            (np.float64(420.0) + np.log10((x + np.float64(0.01)) / np.float64(0.19)) * np.float64(261.5))
            / np.float64(1023.0)
        ),
    )


def _slog3_decode(values: np.ndarray) -> np.ndarray:
    return _piecewise(
        values,
        _SLOG3_CODE_CUT,
        lambda e: (
            (e * np.float64(1023.0) - np.float64(95.0))
            * np.float64(0.01125)
            / (np.float64(171.2102946929) - np.float64(95.0))
        ),
        lambda e: (
            np.float64(10.0) ** ((e * np.float64(1023.0) - np.float64(420.0)) / np.float64(261.5)) * np.float64(0.19)
            - np.float64(0.01)
        ),
    )


def _logc4_encode(values: np.ndarray) -> np.ndarray:
    return _piecewise(
        values,
        _LOGC4_T,
        lambda x: (x - _LOGC4_T) / _LOGC4_S,
        lambda x: (
            ((np.log2(_LOGC4_A * x + np.float64(64.0)) - np.float64(6.0)) / np.float64(14.0)) * _LOGC4_B + _LOGC4_C
        ),
    )


def _logc4_decode(values: np.ndarray) -> np.ndarray:
    return _piecewise(
        values,
        np.float64(0.0),
        lambda e: e * _LOGC4_S + _LOGC4_T,
        lambda e: (
            (np.float64(2.0) ** (np.float64(14.0) * (e - _LOGC4_C) / _LOGC4_B + np.float64(6.0)) - np.float64(64.0))
            / _LOGC4_A
        ),
    )


def _mirror(values: np.ndarray, function: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    return np.copysign(function(np.abs(values)), values)


def _public_target(linear: np.ndarray, gamma: str) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.repeat(linear.astype(np.float32)[:, None], 3, axis=1)[None, :, :]
    source = px.io.from_array(cp.asarray(rgb), colorspace="ACEScg", gamma="linear", channels="RGB")
    encoded = px.color.linear_to_gamma(source, gamma=gamma)
    round_trip = px.color.gamma_to_linear(encoded, gamma=gamma)
    return (
        px.io.to_array(encoded).get()[0, :, 0].astype(np.float64),
        px.io.to_array(round_trip).get()[0, :, 0].astype(np.float64),
    )


def _public_encode_rgb(linear: np.ndarray, gamma: str) -> np.ndarray:
    source = px.io.from_array(
        cp.asarray(linear.astype(np.float32)[None, :, :]),
        colorspace="ACEScg",
        gamma="linear",
        channels="RGB",
    )
    encoded = px.color.linear_to_gamma(source, gamma=gamma)
    return px.io.to_array(encoded).get()[0].astype(np.float64)


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
        draw.text((x + 3, bottom + 3), label, fill=_TEXT, font=font)


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
    rgb_samples = np.repeat(samples[:, None], 3, axis=1) if samples.ndim == 1 else samples
    rgb = np.repeat(rgb_samples[None, :, :], 32, axis=0)
    resized = Image.fromarray(rgb, mode="RGB").resize((_PLOT_WIDTH, 32), resample=Image.Resampling.NEAREST)
    image.paste(resized, (_LEFT, top))


def _sheet(
    gamma: str,
    direct_encode: Callable[[np.ndarray], np.ndarray],
    direct_decode: Callable[[np.ndarray], np.ndarray],
    markers: Sequence[tuple[float, str]],
) -> Image.Image:
    linear = np.linspace(-0.3, 1.5, _SAMPLES, dtype=np.float32).astype(np.float64)
    target_encoded, target_round_trip = _public_target(linear, gamma)
    mirror_encoded = _mirror(linear, direct_encode)
    mirror_round_trip = _mirror(mirror_encoded, direct_decode)
    color_linear = np.stack((linear, linear + np.float64(0.05), linear + np.float64(0.10)), axis=1)
    mirror_color = _mirror(color_linear, direct_encode)
    target_color = _public_encode_rgb(color_linear, gamma)

    image = Image.new("RGB", (_WIDTH, _HEIGHT), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text(
        (_LEFT, 18), f"{gamma}: sign/magnitude mirror vs vendor piecewise signed extension", fill=_TEXT, font=font
    )
    draw.text((_LEFT, 38), "old mirror", fill=_OLD, font=font)
    draw.text((_LEFT + 88, 38), "target", fill=_TARGET, font=font)
    draw.text((_LEFT + 150, 38), "target - old", fill=_DIFFERENCE, font=font)

    encode_lower = float(min(mirror_encoded.min(), target_encoded.min()))
    encode_upper = float(max(mirror_encoded.max(), target_encoded.max()))
    margin = (encode_upper - encode_lower) * 0.04
    encode_lower -= margin
    encode_upper += margin

    draw.text((12, 89), "old code", fill=_OLD, font=font)
    draw.text((12, 128), "target", fill=_TARGET, font=font)
    _draw_strip(image, mirror_color, top=78, lower=encode_lower, upper=encode_upper)
    _draw_strip(image, target_color, top=117, lower=encode_lower, upper=encode_upper)
    _draw_markers(draw, font, linear, markers, 74, 149)

    encode_top, encode_bottom = 190, 430
    draw.rectangle((_LEFT, encode_top, _LEFT + _PLOT_WIDTH, encode_bottom), outline=_GRID)
    draw.text((12, encode_top), "encode", fill=_TEXT, font=font)
    _draw_curve(
        draw,
        linear,
        mirror_encoded,
        top=encode_top,
        bottom=encode_bottom,
        lower=encode_lower,
        upper=encode_upper,
        color=_OLD,
    )
    _draw_curve(
        draw,
        linear,
        target_encoded,
        top=encode_top,
        bottom=encode_bottom,
        lower=encode_lower,
        upper=encode_upper,
        color=_TARGET,
    )
    _draw_markers(draw, font, linear, markers, encode_top, encode_bottom)

    difference = target_encoded - mirror_encoded
    difference_limit = max(abs(float(difference.min())), abs(float(difference.max())), 1e-6)
    difference_top, difference_bottom = 468, 542
    draw.rectangle((_LEFT, difference_top, _LEFT + _PLOT_WIDTH, difference_bottom), outline=_GRID)
    draw.text((12, difference_top), "delta", fill=_DIFFERENCE, font=font)
    _draw_curve(
        draw,
        linear,
        difference,
        top=difference_top,
        bottom=difference_bottom,
        lower=-difference_limit,
        upper=difference_limit,
        color=_DIFFERENCE,
    )
    _draw_markers(draw, font, linear, markers, difference_top, difference_bottom)

    old_error = mirror_round_trip - linear
    target_error = target_round_trip - linear
    error_limit = max(
        abs(float(old_error.min())),
        abs(float(old_error.max())),
        abs(float(target_error.min())),
        abs(float(target_error.max())),
        1e-7,
    )
    error_top, error_bottom = 574, 648
    draw.rectangle((_LEFT, error_top, _LEFT + _PLOT_WIDTH, error_bottom), outline=_GRID)
    _draw_curve(
        draw,
        linear,
        old_error,
        top=error_top,
        bottom=error_bottom,
        lower=-error_limit,
        upper=error_limit,
        color=_OLD,
    )
    _draw_curve(
        draw,
        linear,
        target_error,
        top=error_top,
        bottom=error_bottom,
        lower=-error_limit,
        upper=error_limit,
        color=_TARGET,
    )
    draw.text((12, error_top), "round-trip", fill=_TEXT, font=font)
    draw.text(
        (_LEFT + 5, error_top + 4),
        "old: float64 mirror oracle | target: GPU float32 target",
        fill=_TEXT,
        font=font,
    )
    _draw_markers(draw, font, linear, markers, error_top, error_bottom)
    return image


def generate_sheets(directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    slog3_path = directory / "slog3-negative-extension.png"
    logc4_path = directory / "logc4-negative-extension.png"
    _sheet(
        "S-Log3",
        _slog3_encode,
        _slog3_decode,
        ((-0.25, "negative"), (0.0, "0"), (0.01125, "cut"), (0.18, "18%"), (0.9, "90%"), (1.5, "overshoot")),
    ).save(slog3_path, format="PNG", compress_level=9)
    _sheet(
        "ARRI-LogC4",
        _logc4_encode,
        _logc4_decode,
        ((-0.25, "negative"), (float(_LOGC4_T), "t"), (0.0, "0"), (0.18, "18%"), (1.0, "1"), (1.5, "overshoot")),
    ).save(logc4_path, format="PNG", compress_level=9)
    return slog3_path, logc4_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    arguments = parser.parse_args()
    generate_sheets(arguments.output_directory)


if __name__ == "__main__":
    main()
