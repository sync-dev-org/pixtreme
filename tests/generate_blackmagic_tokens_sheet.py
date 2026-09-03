"""Generate deterministic visual comparisons for v1-blackmagic-tokens acceptance 52."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import cupy as cp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import pixtreme as px

_WIDTH = 1200
_LEFT = 126
_RIGHT = 30
_PLOT_WIDTH = _WIDTH - _LEFT - _RIGHT
_SAMPLES = 2048
_BACKGROUND = (17, 20, 27)
_GRID = (55, 62, 76)
_TEXT = (225, 229, 238)
_FILM = (72, 207, 205)
_DAVINCI = (247, 197, 72)
_ERROR = (243, 101, 128)

_A = np.float64("0.08692876065491224")
_B = np.float64("0.005494072432257808")
_C = np.float64("0.5300133392291939")
_D = np.float64("8.283605932402494")
_E = np.float64("0.09246575342465753")
_FILM_CUT = np.float64("0.005")

_DI_A = np.float64("0.0075")
_DI_B = np.float64("7.0")
_DI_C = np.float64("0.07329248")
_DI_M = np.float64("10.44426855")
_DI_CUT = np.float64("0.00262409")


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


def _film_encode(values: np.ndarray) -> np.ndarray:
    return _piecewise(values, values < _FILM_CUT, lambda part: _D * part + _E, lambda part: _A * np.log(part + _B) + _C)


def _davinci_encode(values: np.ndarray) -> np.ndarray:
    return _piecewise(
        values,
        values <= _DI_CUT,
        lambda part: _DI_M * part,
        lambda part: (np.log2(part + _DI_A) + _DI_B) * _DI_C,
    )


def _public_curve(linear: np.ndarray, gamma: str) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.repeat(linear.astype(np.float32)[:, None], 3, axis=1)[None, :, :]
    source = px.io.from_array(cp.asarray(rgb), colorspace="ACEScg", gamma="linear", channels="RGB")
    encoded = px.color.linear_to_gamma(source, gamma=gamma)
    restored = px.color.gamma_to_linear(encoded, gamma=gamma)
    return px.io.to_array(encoded).get()[0, :, 0], px.io.to_array(restored).get()[0, :, 0]


def _draw_curve(
    draw: ImageDraw.ImageDraw,
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    top: int,
    bottom: int,
    lower: float,
    upper: float,
    color: tuple[int, int, int],
) -> None:
    normalized = np.clip((y_values - lower) / (upper - lower), 0.0, 1.0)
    x = np.linspace(_LEFT, _LEFT + _PLOT_WIDTH - 1, x_values.size)
    y = bottom - normalized * (bottom - top)
    draw.line(tuple(zip(x.tolist(), y.tolist(), strict=True)), fill=color, width=2)


def _x_pixel(values: np.ndarray, value: float, *, logarithmic: bool = False) -> int:
    lower = float(values[0])
    upper = float(values[-1])
    if logarithmic:
        fraction = np.log(value / lower) / np.log(upper / lower)
    else:
        fraction = (value - lower) / (upper - lower)
    return _LEFT + int(round(fraction * (_PLOT_WIDTH - 1)))


def _markers(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    values: np.ndarray,
    top: int,
    bottom: int,
    markers: tuple[tuple[float, str], ...],
    *,
    logarithmic: bool = False,
    label_rows: int = 1,
    show_labels: bool = True,
) -> None:
    for index, (value, label) in enumerate(markers):
        x = _x_pixel(values, value, logarithmic=logarithmic)
        draw.line((x, top, x, bottom), fill=_GRID, width=1)
        if show_labels:
            bounds = draw.textbbox((0, 0), label, font=font)
            label_x = min(x + 3, _WIDTH - _RIGHT - (bounds[2] - bounds[0]))
            draw.text((label_x, bottom + 3 + 12 * (index % label_rows)), label, fill=_TEXT, font=font)


def _curve_sheet() -> Image.Image:
    linear = np.linspace(-0.25, 1.5, _SAMPLES, dtype=np.float32)
    film, film_round_trip = _public_curve(linear, "Blackmagic-Film-Gen-5")
    davinci, davinci_round_trip = _public_curve(linear, "DaVinci-Intermediate")
    film_oracle = _film_encode(linear.astype(np.float64))
    davinci_oracle = _davinci_encode(linear.astype(np.float64))

    cut_linear = np.linspace(-0.02, 0.012, _SAMPLES, dtype=np.float32)
    cut_film, _ = _public_curve(cut_linear, "Blackmagic-Film-Gen-5")
    cut_davinci, _ = _public_curve(cut_linear, "DaVinci-Intermediate")
    cut_film_oracle = _film_encode(cut_linear.astype(np.float64))
    cut_davinci_oracle = _davinci_encode(cut_linear.astype(np.float64))

    high_linear = np.geomspace(0.18, 222.86, _SAMPLES, dtype=np.float32)
    high_film, _ = _public_curve(high_linear, "Blackmagic-Film-Gen-5")
    high_davinci, _ = _public_curve(high_linear, "DaVinci-Intermediate")
    high_film_oracle = _film_encode(high_linear.astype(np.float64))
    high_davinci_oracle = _davinci_encode(high_linear.astype(np.float64))

    for actual, oracle in (
        (film, film_oracle),
        (davinci, davinci_oracle),
        (cut_film, cut_film_oracle),
        (cut_davinci, cut_davinci_oracle),
        (high_film, high_film_oracle),
        (high_davinci, high_davinci_oracle),
    ):
        np.testing.assert_allclose(actual, oracle, rtol=0.0, atol=2e-6)
    np.testing.assert_allclose(film_round_trip, linear, rtol=0.0, atol=1.2e-5)
    np.testing.assert_allclose(davinci_round_trip, linear, rtol=0.0, atol=1.2e-5)

    image = Image.new("RGB", (_WIDTH, 1080), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text(
        (_LEFT, 18), "Blackmagic Film Gen 5 / DaVinci Intermediate signed transfer comparison", fill=_TEXT, font=font
    )
    draw.text((_LEFT, 38), "Film Gen 5", fill=_FILM, font=font)
    draw.text((_LEFT + 100, 38), "DaVinci Intermediate", fill=_DAVINCI, font=font)
    overview_markers = ((-0.25, "negative -0.25"), (0.18, "both 0.18"), (1.0, "both 1"), (1.5, "overshoot 1.5"))
    lower = float(min(film_oracle.min(), davinci_oracle.min()))
    upper = float(max(film_oracle.max(), davinci_oracle.max()))
    top, bottom = 78, 300
    draw.rectangle((_LEFT, top, _LEFT + _PLOT_WIDTH, bottom), outline=_GRID)
    _draw_curve(draw, linear, film_oracle, top=top, bottom=bottom, lower=lower, upper=upper, color=_FILM)
    _draw_curve(draw, linear, davinci_oracle, top=top, bottom=bottom, lower=lower, upper=upper, color=_DAVINCI)
    draw.text((12, top), "signed overview", fill=_TEXT, font=font)
    draw.text((12, top + 12), "linear x", fill=_TEXT, font=font)
    _markers(draw, font, linear, top, bottom, overview_markers)

    cut_markers = (
        (-0.01, "DaVinci -0.01"),
        (0.0, "both 0"),
        (float(_DI_CUT), "DaVinci cut 0.00262409"),
        (float(_FILM_CUT), "Film cut 0.005"),
    )
    lower = float(min(cut_film_oracle.min(), cut_davinci_oracle.min()))
    upper = float(max(cut_film_oracle.max(), cut_davinci_oracle.max()))
    top, bottom = 360, 520
    draw.rectangle((_LEFT, top, _LEFT + _PLOT_WIDTH, bottom), outline=_GRID)
    _draw_curve(draw, cut_linear, cut_film_oracle, top=top, bottom=bottom, lower=lower, upper=upper, color=_FILM)
    _draw_curve(draw, cut_linear, cut_davinci_oracle, top=top, bottom=bottom, lower=lower, upper=upper, color=_DAVINCI)
    draw.text((12, top), "cut detail", fill=_TEXT, font=font)
    draw.text((12, top + 12), "linear x", fill=_TEXT, font=font)
    _markers(draw, font, cut_linear, top, bottom, cut_markers, label_rows=2)

    high_markers = (
        (0.18, "both 0.18"),
        (1.0, "both 1"),
        (10.0, "both 10"),
        (40.0, "both 40"),
        (100.0, "both 100"),
        (222.86, "Film 222.86"),
    )
    lower = float(min(high_film_oracle.min(), high_davinci_oracle.min()))
    upper = float(max(high_film_oracle.max(), high_davinci_oracle.max()))
    top, bottom = 600, 790
    draw.rectangle((_LEFT, top, _LEFT + _PLOT_WIDTH, bottom), outline=_GRID)
    _draw_curve(draw, high_linear, high_film_oracle, top=top, bottom=bottom, lower=lower, upper=upper, color=_FILM)
    _draw_curve(
        draw, high_linear, high_davinci_oracle, top=top, bottom=bottom, lower=lower, upper=upper, color=_DAVINCI
    )
    draw.text((12, top), "public anchors", fill=_TEXT, font=font)
    draw.text((12, top + 12), "log x", fill=_TEXT, font=font)
    _markers(draw, font, high_linear, top, bottom, high_markers, logarithmic=True, label_rows=2)

    for index, (label, error, color) in enumerate(
        (
            ("Film round-trip", film_round_trip - linear, _FILM),
            ("DaVinci round-trip", davinci_round_trip - linear, _DAVINCI),
        )
    ):
        row_top = 880 + index * 100
        row_bottom = row_top + 50
        limit = max(float(np.max(np.abs(error))), 1e-7)
        draw.rectangle((_LEFT, row_top, _LEFT + _PLOT_WIDTH, row_bottom), outline=_GRID)
        draw.text((12, row_top), label, fill=color, font=font)
        _draw_curve(draw, linear, error, top=row_top, bottom=row_bottom, lower=-limit, upper=limit, color=color)
        _markers(draw, font, linear, row_top, row_bottom, overview_markers, show_labels=False)
    return image


def _gamut_target(source_rgb: np.ndarray, colorspace: str) -> np.ndarray:
    frame = px.io.from_array(cp.asarray(source_rgb[None, :, :]), colorspace=colorspace, gamma="linear", channels="RGB")
    return px.io.to_array(px.color.rgb_to_rgb(frame, output_colorspace="Rec.709")).get()[0]


def _strip(values: np.ndarray, *, lower: float, upper: float, height: int = 74) -> Image.Image:
    normalized = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    pixels = np.rint(normalized * 255.0).astype(np.uint8)
    return Image.fromarray(pixels[None, :, :], mode="RGB").resize((_PLOT_WIDTH, height), Image.Resampling.NEAREST)


def _gamut_sheet() -> Image.Image:
    x = np.linspace(-0.25, 1.5, _SAMPLES, dtype=np.float32)
    source = np.stack(
        (x, np.float32(0.45) + np.float32(0.65) * np.sin(np.float32(4.0) * x), np.float32(1.25) - np.float32(0.75) * x),
        axis=1,
    )
    film_gamut = _gamut_target(source, "Blackmagic-Wide-Gamut-Gen-5")
    davinci_gamut = _gamut_target(source, "DaVinci-Wide-Gamut")
    if not np.isfinite(film_gamut).all() or not np.isfinite(davinci_gamut).all():
        raise AssertionError("Blackmagic gamut comparison produced a non-finite value")
    image = Image.new("RGB", (_WIDTH, 500), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((_LEFT, 18), "Blackmagic Wide Gamut Gen 5 / DaVinci Wide Gamut -> Rec.709 linear", fill=_TEXT, font=font)
    lower = float(min(source.min(), film_gamut.min(), davinci_gamut.min()))
    upper = float(max(source.max(), film_gamut.max(), davinci_gamut.max()))
    for index, (label, values) in enumerate(
        (("source", source), ("Blackmagic", film_gamut), ("DaVinci", davinci_gamut))
    ):
        top = 72 + index * 105
        draw.text((12, top + 28), label, fill=_TEXT, font=font)
        image.paste(_strip(values, lower=lower, upper=upper), (_LEFT, top))
    difference = np.abs(film_gamut.astype(np.float64) - davinci_gamut.astype(np.float64))
    limit = max(float(difference.max()), 1e-7)
    draw.text((12, 400), "abs delta", fill=_ERROR, font=font)
    image.paste(_strip(difference, lower=0.0, upper=limit, height=42), (_LEFT, 390))
    draw.text((_LEFT, 448), f"max abs delta={limit:.8f}; common source and display scale", fill=_TEXT, font=font)
    return image


def generate_sheets(directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    curve_path = directory / "blackmagic-transfer-curves.png"
    gamut_path = directory / "blackmagic-gamut-comparison.png"
    _curve_sheet().save(curve_path, format="PNG", compress_level=9)
    _gamut_sheet().save(gamut_path, format="PNG", compress_level=9)
    return curve_path, gamut_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    arguments = parser.parse_args()
    generate_sheets(arguments.output_directory)


if __name__ == "__main__":
    main()
