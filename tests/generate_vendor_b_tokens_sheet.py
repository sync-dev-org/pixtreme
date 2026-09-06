"""Generate deterministic visual-acceptance sheets for the vendor-B token group."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import pixtreme as px

_WIDTH = 1800
_BACKGROUND = (17, 20, 27)
_PANEL = (27, 32, 42)
_GRID = (61, 70, 87)
_TEXT = (225, 229, 238)
_MUTED = (155, 164, 182)
_GPU = (72, 207, 205)
_CURVE_COLORS = ((72, 207, 205), (247, 197, 72), (175, 126, 242), (244, 116, 146))


@dataclass(frozen=True)
class _Curve:
    gamma: str
    encode_cut: float
    decode_cut: float
    encode_window: float
    decode_window: float


_CURVES = (
    _Curve("N-Log", 0.3784157394368526, 0.4625960144726521, 0.04, 0.02),
    _Curve("L-Log", 0.006, 0.1371004734320989, 0.0015, 0.004),
    _Curve("Apple-Log", 0.01, 0.20855531595464208, 0.003, 0.008),
    _Curve("Samsung-Log", 0.01, 0.20656190889447099, 0.003, 0.008),
)


def _frame(values: np.ndarray, *, colorspace: str = "ACEScg", gamma: str = "linear") -> px.core.Frame:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = np.repeat(array[:, None], 3, axis=1)[None]
    elif array.ndim == 2 and array.shape[-1] == 3:
        array = array[None]
    return px.io.from_array(cp.asarray(array), colorspace=colorspace, gamma=gamma, channels="RGB")


def _gpu_encode(values: np.ndarray, gamma: str) -> np.ndarray:
    result = px.color.linear_to_gamma(_frame(values), gamma=gamma)
    return px.io.to_array(result).get()[0, :, 0]


def _gpu_decode(values: np.ndarray, gamma: str) -> np.ndarray:
    result = px.color.gamma_to_linear(_frame(values, gamma=gamma), gamma=gamma)
    return px.io.to_array(result).get()[0, :, 0]


def _encode(values: np.ndarray, gamma: str, *, printed: bool = False) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    result = np.empty_like(source)
    if gamma == "N-Log":
        lower = source < (0.328 if printed else 0.3784157394368526)
        result[lower] = 650.0 * np.cbrt(source[lower] + 0.0075) / 1023.0
        result[~lower] = (150.0 * np.log(source[~lower]) + 619.0) / 1023.0
    elif gamma == "L-Log":
        lower = source < 0.006
        if printed:
            result[lower] = 8.0 * source[lower] + 0.09
        else:
            result[lower] = 7.898308971401108 * source[lower] + 0.08971061960369227
        result[~lower] = 0.27 * np.log10(1.3 * source[~lower] + 0.0115) + 0.6
    elif gamma == "Apple-Log":
        collapsed = source < -0.05641088
        quadratic = (~collapsed) & (source < 0.01)
        logarithmic = ~(collapsed | quadratic)
        result[collapsed] = 0.0
        result[quadratic] = 47.28711236 * (source[quadratic] + 0.05641088) ** 2
        result[logarithmic] = 0.08550479 * np.log2(source[logarithmic] + 0.00964052) + 0.69336945
    else:
        lower = source < 0.01
        g2 = -0.24597 if printed else -0.245973605190997
        result[lower] = -0.20942 * np.log10(0.016904 - source[lower]) + g2
        result[~lower] = 0.258984868 * np.log10(source[~lower] + 0.0003645) + 0.720504856
    return result


def _decode(values: np.ndarray, gamma: str, *, printed: bool = False) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    result = np.empty_like(source)
    if gamma == "N-Log":
        lower = source < (452.0 / 1023.0 if printed else 0.4625960144726521)
        result[lower] = (source[lower] * 1023.0 / 650.0) ** 3 - 0.0075
        result[~lower] = np.exp((source[~lower] * 1023.0 - 619.0) / 150.0)
    elif gamma == "L-Log":
        lower = source < (0.1380 if printed else 0.1371004734320989)
        if printed:
            result[lower] = (source[lower] - 0.09) / 8.0
        else:
            result[lower] = (source[lower] - 0.08971061960369227) / 7.898308971401108
        result[~lower] = (10.0 ** ((source[~lower] - 0.6) / 0.27) - 0.0115) / 1.3
    elif gamma == "Apple-Log":
        collapsed = source < 0.0
        square_root = (~collapsed) & (source < 0.20855531595464208)
        exponential = ~(collapsed | square_root)
        result[collapsed] = -0.05641088
        result[square_root] = np.sqrt(source[square_root] / 47.28711236) - 0.05641088
        result[exponential] = 2.0 ** ((source[exponential] - 0.69336945) / 0.08550479) - 0.00964052
    else:
        lower = source < (0.206561909 if printed else 0.20656190889447099)
        g2 = -0.24597 if printed else -0.245973605190997
        result[lower] = 0.016904 - 10.0 ** ((source[lower] - g2) / -0.20942)
        result[~lower] = 10.0 ** ((source[~lower] - 0.720504856) / 0.258984868) - 0.0003645
    return result


def _panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=12, fill=_PANEL)
    draw.text((left + 16, top + 12), title, fill=_TEXT, font=ImageFont.load_default())
    plot = (left + 54, top + 44, right - 16, bottom - 34)
    _grid(draw, plot)
    return plot


def _grid(draw: ImageDraw.ImageDraw, plot: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = plot
    for fraction in np.linspace(0.0, 1.0, 5):
        x = round(x0 + fraction * (x1 - x0))
        y = round(y0 + fraction * (y1 - y0))
        draw.line((x, y0, x, y1), fill=_GRID)
        draw.line((x0, y, x1, y), fill=_GRID)


def _sub_panels(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
) -> tuple[tuple[int, int, int, int], ...]:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=12, fill=_PANEL)
    draw.text((left + 16, top + 12), title, fill=_TEXT, font=ImageFont.load_default())
    gap = 10
    width = (right - left - 32 - 3 * gap) // 4
    plots = []
    for index, curve in enumerate(_CURVES):
        column = left + 16 + index * (width + gap)
        plot = (column + 8, top + 60, column + width - 8, bottom - 38)
        draw.text((column + 8, top + 38), curve.gamma, fill=_CURVE_COLORS[index], font=ImageFont.load_default())
        _grid(draw, plot)
        plots.append(plot)
    return tuple(plots)


def _points(
    x: np.ndarray,
    y: np.ndarray,
    plot: tuple[int, int, int, int],
    *,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> list[tuple[int, int]]:
    x0, y0, x1, y1 = plot
    xmin, xmax = (float(np.min(x)), float(np.max(x))) if x_range is None else x_range
    ymin, ymax = (float(np.min(y)), float(np.max(y))) if y_range is None else y_range
    horizontal = x0 + (np.asarray(x, dtype=np.float64) - xmin) * (x1 - x0) / (xmax - xmin)
    vertical = y1 - (np.asarray(y, dtype=np.float64) - ymin) * (y1 - y0) / (ymax - ymin)
    return [(round(px_value), round(py_value)) for px_value, py_value in zip(horizontal, vertical, strict=True)]


def _vertical_marker(draw: ImageDraw.ImageDraw, plot: tuple[int, int, int, int], fraction: float) -> None:
    x0, y0, x1, y1 = plot
    x = round(x0 + fraction * (x1 - x0))
    for y in range(y0, y1, 6):
        draw.line((x, y, x, min(y + 3, y1)), fill=_MUTED)


def _legend(draw: ImageDraw.ImageDraw, origin: tuple[int, int]) -> None:
    x, y = origin
    for index, curve in enumerate(_CURVES):
        color = _CURVE_COLORS[index]
        draw.line((x, y + index * 18 + 6, x + 24, y + index * 18 + 6), fill=color, width=3)
        draw.text((x + 32, y + index * 18), curve.gamma, fill=_TEXT, font=ImageFont.load_default())


def _draw_cut_row(
    draw: ImageDraw.ImageDraw,
    plots: tuple[tuple[int, int, int, int], ...],
    *,
    decode: bool,
) -> None:
    for index, curve in enumerate(_CURVES):
        center = curve.decode_cut if decode else curve.encode_cut
        half_width = curve.decode_window if decode else curve.encode_window
        inputs = np.linspace(center - half_width, center + half_width, 513)
        oracle = _decode(inputs, curve.gamma) if decode else _encode(inputs, curve.gamma)
        gpu = (
            _gpu_decode(inputs.astype(np.float32), curve.gamma)
            if decode
            else _gpu_encode(inputs.astype(np.float32), curve.gamma)
        )
        y_pad = max(float(np.ptp(oracle)) * 0.05, 1e-12)
        y_range = (float(np.min(oracle)) - y_pad, float(np.max(oracle)) + y_pad)
        x_range = (float(inputs[0]), float(inputs[-1]))
        plot = plots[index]
        _vertical_marker(draw, plot, 0.5)
        draw.line(_points(inputs, oracle, plot, x_range=x_range, y_range=y_range), fill=_CURVE_COLORS[index], width=2)
        for point in _points(inputs[::24], gpu[::24], plot, x_range=x_range, y_range=y_range):
            draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=_GPU)
        label = "ENC_CUT" if decode else "CUT"
        draw.text(
            (plot[0], plot[3] + 8),
            f"{label}={center:.16g}; window +/-{half_width:.3g}; dashed=cut",
            fill=_MUTED,
            font=ImageFont.load_default(),
        )


def _draw_difference_row(
    draw: ImageDraw.ImageDraw,
    plots: tuple[tuple[int, int, int, int], ...],
    *,
    decode: bool,
) -> None:
    for index, curve in enumerate(_CURVES):
        center = curve.decode_cut if decode else curve.encode_cut
        half_width = curve.decode_window * 2 if decode else curve.encode_window * 2
        inputs = np.linspace(center - half_width, center + half_width, 1025)
        production = _decode(inputs, curve.gamma) if decode else _encode(inputs, curve.gamma)
        printed = _decode(inputs, curve.gamma, printed=True) if decode else _encode(inputs, curve.gamma, printed=True)
        difference = production - printed
        peak = max(float(np.max(np.abs(difference))), 1e-12)
        plot = plots[index]
        draw.line(
            _points(inputs, difference, plot, y_range=(-peak * 1.1, peak * 1.1)),
            fill=_CURVE_COLORS[index],
            width=2,
        )
        note = "published definition; no difference" if curve.gamma == "Apple-Log" else f"max |delta|={peak:.3g}"
        draw.text((plot[0], plot[3] + 8), note, fill=_MUTED, font=ImageFont.load_default())


def _transfer_sheet() -> Image.Image:
    image = Image.new("RGB", (_WIDTH, 1930), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((24, 18), "Vendor B transfers — GPU float32 vs independent host float64", fill=_TEXT, font=font)
    draw.text(
        (24, 40),
        "Reflectance input; per-curve continuity rule; signed extension or Apple collapse; no upper clipping",
        fill=_MUTED,
        font=font,
    )

    full_encode = _panel(draw, (24, 72, 880, 390), "Signed full-domain encode (symlog reflectance axis)")
    full_decode = _panel(draw, (904, 72, 1776, 390), "Signed full-domain decode (symlog output axis)")
    linear = np.concatenate((-np.geomspace(5e-5, 0.5, 512)[::-1], (0.0,), np.geomspace(5e-5, 64.0, 1536)))
    symlog = np.sign(linear) * np.log10(1.0 + 100.0 * np.abs(linear))
    encoded_axis = np.linspace(-0.5, 1.5, 2049)
    encode_values = np.concatenate(tuple(_encode(linear, curve.gamma) for curve in _CURVES))
    encode_range = (float(np.min(encode_values)), float(np.max(encode_values)))
    decode_values = np.concatenate(tuple(_decode(encoded_axis, curve.gamma) for curve in _CURVES))
    decoded_symlog_all = np.sign(decode_values) * np.log10(1.0 + np.abs(decode_values))
    decode_range = (float(np.min(decoded_symlog_all)), float(np.max(decoded_symlog_all)))
    for index, curve in enumerate(_CURVES):
        color = _CURVE_COLORS[index]
        draw.line(_points(symlog, _encode(linear, curve.gamma), full_encode, y_range=encode_range), fill=color, width=2)
        decoded = _decode(encoded_axis, curve.gamma)
        decoded_symlog = np.sign(decoded) * np.log10(1.0 + np.abs(decoded))
        draw.line(_points(encoded_axis, decoded_symlog, full_decode, y_range=decode_range), fill=color, width=2)
    _legend(draw, (88, 94))

    encode_plots = _sub_panels(draw, (24, 414, 1776, 700), "Encode cut enlargement per curve — GPU dots over oracle")
    decode_plots = _sub_panels(draw, (24, 724, 1776, 1010), "Decode cut enlargement per curve — GPU dots over oracle")
    _draw_cut_row(draw, encode_plots, decode=False)
    _draw_cut_row(draw, decode_plots, decode=True)

    encode_difference = _sub_panels(
        draw,
        (24, 1034, 1776, 1300),
        "Production minus printed definition — encode (Apple is definition-identical)",
    )
    decode_difference = _sub_panels(
        draw,
        (24, 1324, 1776, 1590),
        "Production minus printed definition — decode (Apple is definition-identical)",
    )
    _draw_difference_row(draw, encode_difference, decode=False)
    _draw_difference_row(draw, decode_difference, decode=True)

    residual = _panel(draw, (24, 1614, 880, 1888), "Encode/decode round-trip residual across signed domain")
    for index, curve in enumerate(_CURVES):
        values = np.linspace(-0.5, 64.0, 4097, dtype=np.float64).astype(np.float32)
        if curve.gamma == "Apple-Log":
            values = values[values >= np.float32(-0.05641088)]
        restored = _gpu_decode(_gpu_encode(values, curve.gamma), curve.gamma)
        error = restored.astype(np.float64) - values.astype(np.float64)
        draw.line(
            _points(values, error, residual, x_range=(-0.5, 64.0), y_range=(-2e-4, 2e-4)),
            fill=_CURVE_COLORS[index],
            width=2,
        )

    draw.rounded_rectangle((904, 1614, 1776, 1888), radius=12, fill=_PANEL)
    draw.text((920, 1626), "Published anchors — GPU y*1023 / independent oracle", fill=_TEXT, font=font)
    anchors = {
        "N-Log": np.asarray((0.0, 0.18, 0.9, 1.0), dtype=np.float32),
        "L-Log": np.asarray((0.0, 0.02, 0.18, 0.9), dtype=np.float32),
        "Apple-Log": np.asarray((0.0, 0.18, 0.9, 12.0), dtype=np.float32),
        "Samsung-Log": np.asarray((0.0, 0.01, 0.18, 0.9, 12.0), dtype=np.float32),
    }
    for index, curve in enumerate(_CURVES):
        values = anchors[curve.gamma]
        gpu_codes = _gpu_encode(values, curve.gamma).astype(np.float64) * 1023.0
        oracle_codes = _encode(values.astype(np.float64), curve.gamma) * 1023.0
        draw.text((920, 1660 + index * 48), curve.gamma, fill=_CURVE_COLORS[index], font=font)
        draw.text(
            (920, 1678 + index * 48),
            f"GPU {np.array2string(gpu_codes, precision=4)} | oracle {np.array2string(oracle_codes, precision=4)}",
            fill=_MUTED,
            font=font,
        )
    return image


def _display(values: np.ndarray) -> np.ndarray:
    encoded = px.color.linear_to_gamma(_frame(values.reshape(-1, 3), colorspace="Rec.709"), gamma="sRGB")
    array = px.io.to_array(encoded).get()[0].reshape(values.shape)
    return np.asarray(np.clip(array, 0.0, 1.0) * 255.0 + 0.5, dtype=np.uint8)


def _composite_sheet() -> Image.Image:
    image = Image.new("RGB", (_WIDTH, 930), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((24, 18), "Vendor B independent colorspace + gamma composites to Rec.709 / linear", fill=_TEXT, font=font)
    draw.text(
        (24, 40),
        "Apple-Wide-Gamut uses R(0.725,0.301) G(0.221,0.814) B(0.068,-0.076), D65, native row, Bradford",
        fill=_MUTED,
        font=font,
    )
    draw.text(
        (24, 58),
        "CAT02 is auxiliary only; Apple Log 2 = Apple-Wide-Gamut + Apple-Log; Nikon/Leica/Samsung pairs use Rec.2020",
        fill=_MUTED,
        font=font,
    )
    width, height = 1220, 140
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    source = np.stack(
        (1.35 * np.broadcast_to(x, (height, width)) - 0.1, np.broadcast_to(y, (height, width)), 1.3 - x - 0.45 * y),
        axis=-1,
    )
    cases = (
        ("Rec.2020", "N-Log"),
        ("Rec.2020", "L-Log"),
        ("Rec.2020", "Apple-Log"),
        ("Rec.2020", "Samsung-Log"),
        ("Apple-Wide-Gamut", "Apple-Log"),
    )
    for index, (colorspace, gamma) in enumerate(cases):
        linear = _frame(source.reshape(-1, 3), colorspace=colorspace)
        encoded = px.color.linear_to_gamma(linear, gamma=gamma)
        converted = px.color.rgb_to_rgb(encoded, output_colorspace="Rec.709", output_gamma="linear")
        rec709 = px.io.to_array(converted).get()[0].reshape(source.shape)
        strip = Image.fromarray(_display(rec709), mode="RGB")
        top = 104 + index * 160
        image.paste(strip, (24, top))
        draw.text((1264, top + 20), f"{colorspace} + {gamma}", fill=_CURVE_COLORS[index % 4], font=font)
        draw.text((1264, top + 44), "decode transfer -> gamut matrix -> Rec.709", fill=_TEXT, font=font)
        draw.text((1264, top + 68), "negative / 18% / >1 source values included", fill=_MUTED, font=font)
    return image


def generate(directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    transfer_path = directory / "vendor-b-transfers.png"
    composite_path = directory / "vendor-b-composites.png"
    _transfer_sheet().save(transfer_path, format="PNG", optimize=False, compress_level=9)
    _composite_sheet().save(composite_path, format="PNG", optimize=False, compress_level=9)
    return transfer_path, composite_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    for path in generate(args.directory):
        print(path)


if __name__ == "__main__":
    main()
