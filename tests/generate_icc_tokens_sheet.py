"""Generate deterministic visual evidence for v1-io-icc acceptance 24."""

from __future__ import annotations

import argparse
from pathlib import Path

import cupy as cp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import pixtreme as px

_WIDTH = 1440
_LEFT = 150
_RIGHT = 36
_BACKGROUND = (17, 20, 27)
_GRID = (55, 62, 76)
_TEXT = (225, 229, 238)
_GPU = (72, 207, 205)
_ORACLE = (247, 197, 72)
_ACCENT = (175, 126, 242)
_ERROR = (243, 101, 128)
_SAMPLES = 4096
_D65 = (0.3127, 0.3290)
_D50 = (0.3457, 0.3585)
_DEFINITIONS = {
    "sRGB": (((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)), _D65),
    "Adobe-RGB": (((0.6400, 0.3300), (0.2100, 0.7100), (0.1500, 0.0600)), _D65),
    "ProPhoto-RGB": (((0.7347, 0.2653), (0.1596, 0.8404), (0.0366, 0.0001)), _D50),
}
_BRADFORD = np.asarray(
    ((0.8951, 0.2664, -0.1614), (-0.7502, 1.7135, 0.0367), (0.0389, -0.0685, 1.0296)),
    dtype=np.float64,
)


def _frame(values: np.ndarray, *, colorspace: str = "sRGB", gamma: str = "linear") -> px.core.Frame:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = np.repeat(array[:, None], 3, axis=1)
    return px.io.from_array(cp.asarray(array[None]), colorspace=colorspace, gamma=gamma, channels="RGB")


def _gpu_encode(values: np.ndarray, gamma: str) -> np.ndarray:
    return px.color.linear_to_gamma(_frame(values), gamma=gamma).data.get()[0, :, 0].astype(np.float64)


def _gpu_decode(values: np.ndarray, gamma: str) -> np.ndarray:
    return px.color.gamma_to_linear(_frame(values, gamma=gamma), gamma=gamma).data.get()[0, :, 0].astype(np.float64)


def _signed_power(values: np.ndarray, exponent: float) -> np.ndarray:
    return np.copysign(np.abs(np.asarray(values, dtype=np.float64)) ** exponent, values)


def _encode(values: np.ndarray, gamma: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if gamma == "Gamma-1.8":
        return _signed_power(values, 1.0 / 1.8)
    if gamma == "Adobe-RGB":
        return _signed_power(values, 256.0 / 563.0)
    if gamma == "ProPhoto-RGB":
        result = np.empty_like(values)
        lower = values < 1.0 / 512.0
        result[lower] = 16.0 * values[lower]
        result[~lower] = values[~lower] ** (1.0 / 1.8)
        return result
    if gamma == "sRGB":
        result = np.empty_like(values)
        lower = values <= 0.0031308
        result[lower] = 12.92 * values[lower]
        result[~lower] = 1.055 * values[~lower] ** (1.0 / 2.4) - 0.055
        return result
    raise ValueError(gamma)


def _decode(values: np.ndarray, gamma: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if gamma == "Gamma-1.8":
        return _signed_power(values, 1.8)
    if gamma == "Adobe-RGB":
        return _signed_power(values, 563.0 / 256.0)
    if gamma == "ProPhoto-RGB":
        result = np.empty_like(values)
        lower = values < 1.0 / 32.0
        result[lower] = values[lower] / 16.0
        result[~lower] = values[~lower] ** 1.8
        return result
    if gamma == "sRGB":
        result = np.empty_like(values)
        lower = values <= 0.04045
        result[lower] = values[lower] / 12.92
        result[~lower] = ((values[~lower] + 0.055) / 1.055) ** 2.4
        return result
    raise ValueError(gamma)


def _xy_to_xyz(xy: tuple[float, float]) -> np.ndarray:
    x, y = xy
    return np.asarray((x / y, 1.0, (1.0 - x - y) / y), dtype=np.float64)


def _rgb_to_xyz(definition: tuple[tuple[tuple[float, float], ...], tuple[float, float]]) -> np.ndarray:
    primaries, white = definition
    unscaled = np.asarray(
        (tuple(x / y for x, y in primaries), (1.0, 1.0, 1.0), tuple((1.0 - x - y) / y for x, y in primaries)),
        dtype=np.float64,
    )
    return unscaled @ np.diag(np.linalg.solve(unscaled, _xy_to_xyz(white)))


def _conversion(source: str, target: str) -> np.ndarray:
    source_definition = _DEFINITIONS[source]
    target_definition = _DEFINITIONS[target]
    source_cones = _BRADFORD @ _xy_to_xyz(source_definition[1])
    target_cones = _BRADFORD @ _xy_to_xyz(target_definition[1])
    adaptation = np.linalg.inv(_BRADFORD) @ np.diag(target_cones / source_cones) @ _BRADFORD
    return np.linalg.inv(_rgb_to_xyz(target_definition)) @ adaptation @ _rgb_to_xyz(source_definition)


def _coordinates(values: np.ndarray, lower: float, upper: float, start: int, extent: int) -> np.ndarray:
    return start + np.rint((values - lower) / (upper - lower) * extent).astype(np.int32)


def _panel(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    *,
    box: tuple[int, int, int, int],
    title: str,
    x: np.ndarray,
    curves: tuple[tuple[np.ndarray, tuple[int, int, int]], ...],
) -> None:
    left, top, width, height = box
    finite = np.concatenate(tuple(values[np.isfinite(values)] for values, _ in curves))
    lower, upper = float(finite.min()), float(finite.max())
    padding = max((upper - lower) * 0.05, 1e-12)
    lower -= padding
    upper += padding
    draw.rectangle((left, top, left + width, top + height), outline=_GRID)
    draw.text((left, top - 21), title, fill=_TEXT, font=font)
    xp = _coordinates(np.asarray(x, dtype=np.float64), float(x[0]), float(x[-1]), left, width)
    for values, color in curves:
        yp = top + height - _coordinates(np.asarray(values), lower, upper, 0, height)
        draw.line(tuple(zip(xp, yp, strict=True)), fill=color, width=2)
    draw.text((left, top + height + 4), f"x {float(x[0]):.7g} .. {float(x[-1]):.7g}", fill=_TEXT, font=font)
    draw.text((left + width - 230, top + height + 4), f"y {lower:.7g} .. {upper:.7g}", fill=_TEXT, font=font)


def _transfer_sheet() -> Image.Image:
    image = Image.new("RGB", (_WIDTH, 1740), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width = (_WIDTH - _LEFT - _RIGHT - 30) // 2
    draw.text((_LEFT, 18), "ICC transfer tokens: GPU float32 vs independent host float64", fill=_TEXT, font=font)
    draw.text((_LEFT, 40), "GPU", fill=_GPU, font=font)
    draw.text((_LEFT + 50, 40), "oracle", fill=_ORACLE, font=font)
    signed = np.linspace(-0.25, 1.5, _SAMPLES, dtype=np.float64).astype(np.float32)

    panels = (
        ("Gamma-1.8 encode", signed, _encode(signed, "Gamma-1.8"), _gpu_encode(signed, "Gamma-1.8")),
        ("Adobe-RGB encode (563/256)", signed, _encode(signed, "Adobe-RGB"), _gpu_encode(signed, "Adobe-RGB")),
        ("ProPhoto-RGB encode", signed, _encode(signed, "ProPhoto-RGB"), _gpu_encode(signed, "ProPhoto-RGB")),
        ("ProPhoto-RGB decode", signed, _decode(signed, "ProPhoto-RGB"), _gpu_decode(signed, "ProPhoto-RGB")),
    )
    for index, (title, x, oracle, gpu) in enumerate(panels):
        row, column = divmod(index, 2)
        _panel(
            draw,
            font,
            box=(_LEFT + column * (width + 30), 90 + row * 300, width, 230),
            title=title,
            x=x,
            curves=((oracle, _ORACLE), (gpu, _GPU)),
        )

    encode_cut = np.linspace(1.0 / 512.0 - 2e-5, 1.0 / 512.0 + 2e-5, _SAMPLES).astype(np.float32)
    decode_cut = np.linspace(1.0 / 32.0 - 2e-4, 1.0 / 32.0 + 2e-4, _SAMPLES).astype(np.float32)
    _panel(
        draw,
        font,
        box=(_LEFT, 690, width, 230),
        title="ProPhoto encode cut x=1/512",
        x=encode_cut,
        curves=((_encode(encode_cut, "ProPhoto-RGB"), _ORACLE), (_gpu_encode(encode_cut, "ProPhoto-RGB"), _GPU)),
    )
    _panel(
        draw,
        font,
        box=(_LEFT + width + 30, 690, width, 230),
        title="ProPhoto decode cut x=1/32",
        x=decode_cut,
        curves=((_decode(decode_cut, "ProPhoto-RGB"), _ORACLE), (_gpu_decode(decode_cut, "ProPhoto-RGB"), _GPU)),
    )

    residuals = tuple(
        (
            np.abs(_gpu_decode(_gpu_encode(signed, gamma), gamma) - signed.astype(np.float64)),
            color,
        )
        for gamma, color in (("Gamma-1.8", _ACCENT), ("Adobe-RGB", _ORACLE), ("ProPhoto-RGB", _GPU))
    )
    _panel(
        draw,
        font,
        box=(_LEFT, 990, width, 230),
        title="Round-trip residual: 1.8 / Adobe / ProPhoto",
        x=signed,
        curves=residuals,
    )
    power_values = np.linspace(0.0, 1.5, _SAMPLES, dtype=np.float32)
    _panel(
        draw,
        font,
        box=(_LEFT + width + 30, 990, width, 230),
        title="Encode comparison: Gamma-1.8 / 2.2 / Adobe / ProPhoto",
        x=power_values,
        curves=(
            (_gpu_encode(power_values, "Gamma-1.8"), _ACCENT),
            (_gpu_encode(power_values, "Gamma-2.2"), _ERROR),
            (_gpu_encode(power_values, "Adobe-RGB"), _ORACLE),
            (_gpu_encode(power_values, "ProPhoto-RGB"), _GPU),
        ),
    )

    anchors = np.asarray((-0.25, 0.0, 1.0 / 512.0, 0.18, 1.0 / 32.0, 1.0, 1.25), dtype=np.float32)
    draw.text((_LEFT, 1305), "Anchors: x | Gamma-1.8 | Adobe-RGB | ProPhoto-RGB (GPU encode)", fill=_TEXT, font=font)
    encoded = {gamma: _gpu_encode(anchors, gamma) for gamma in ("Gamma-1.8", "Adobe-RGB", "ProPhoto-RGB")}
    for index, value in enumerate(anchors):
        draw.text(
            (_LEFT, 1330 + index * 24),
            f"{float(value):>12.7g} | {encoded['Gamma-1.8'][index]:>13.9f} | "
            f"{encoded['Adobe-RGB'][index]:>13.9f} | {encoded['ProPhoto-RGB'][index]:>13.9f}",
            fill=_TEXT,
            font=font,
        )
    draw.text(
        (_LEFT, 1520),
        "Curves include negative values, both ProPhoto branch cuts, 18% gray, 1.0, and overshoot.",
        fill=_TEXT,
        font=font,
    )
    draw.text(
        (_LEFT, 1544),
        "Coincident GPU/oracle lines and smooth residuals expose discontinuity or banding regressions.",
        fill=_TEXT,
        font=font,
    )
    return image


def _preview(values: np.ndarray, height: int) -> np.ndarray:
    return np.repeat(np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)[None], height, axis=0)


def _rgb_sheet() -> Image.Image:
    image = Image.new("RGB", (_WIDTH, 1200), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width = _WIDTH - _LEFT - _RIGHT
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    encoded = np.stack((1.15 * x - 0.08, 0.12 + 0.82 * np.sin(np.pi * x), 1.08 * (1.0 - x) - 0.04), axis=1)
    draw.text((_LEFT, 18), "ICC gamut + transfer to sRGB/sRGB: GPU vs host matrix/TRC oracle", fill=_TEXT, font=font)
    cases = (("Adobe-RGB", "Adobe-RGB"), ("ProPhoto-RGB", "Gamma-1.8"), ("ProPhoto-RGB", "ProPhoto-RGB"))
    for row, (colorspace, gamma) in enumerate(cases):
        source = _frame(encoded, colorspace=colorspace, gamma=gamma)
        actual = px.color.rgb_to_rgb(source, output_colorspace="sRGB", output_gamma="sRGB").data.get()[0]
        linear = _decode(encoded, gamma)
        oracle = _encode(linear @ _conversion(colorspace, "sRGB").T, "sRGB")
        top = 85 + row * 350
        image.paste(Image.fromarray(_preview(actual, 145), mode="RGB"), (_LEFT, top))
        image.paste(Image.fromarray(_preview(oracle, 145), mode="RGB"), (_LEFT, top + 150))
        draw.rectangle((_LEFT, top, _LEFT + width - 1, top + 294), outline=_GRID)
        draw.text((24, top + 65), f"{colorspace}\n{gamma}", fill=_TEXT, font=font)
        draw.text(
            (_LEFT, top + 300),
            f"top GPU / bottom oracle; max abs={np.max(np.abs(actual - oracle)):.3e}",
            fill=_TEXT,
            font=font,
        )
    return image


def generate(directory: Path) -> tuple[Path, Path]:
    """Generate the deterministic transfer and RGB-conversion comparison images."""
    directory.mkdir(parents=True, exist_ok=True)
    transfer = directory / "icc-transfer-curves.png"
    rgb = directory / "icc-rgb-to-rgb.png"
    _transfer_sheet().save(transfer, format="PNG", compress_level=9)
    _rgb_sheet().save(rgb, format="PNG", compress_level=9)
    return transfer, rgb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(".nf/tmp/sheets-39-icc"))
    arguments = parser.parse_args()
    for path in generate(arguments.output_dir):
        print(path)


if __name__ == "__main__":
    main()
