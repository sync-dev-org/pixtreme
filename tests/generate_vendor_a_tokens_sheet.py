"""Generate deterministic visual-acceptance sheets for the vendor-A token group."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import pixtreme as px

_WIDTH = 1600
_BACKGROUND = (17, 20, 27)
_PANEL = (27, 32, 42)
_GRID = (61, 70, 87)
_TEXT = (225, 229, 238)
_MUTED = (155, 164, 182)
_GPU = (72, 207, 205)
_CURVE_COLORS = ((72, 207, 205), (247, 197, 72), (175, 126, 242))


@dataclass(frozen=True)
class _Curve:
    gamma: str
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    x: float
    encoded_cut: float
    printed_x: float
    printed_encoded_cut: float


_CURVES = (
    _Curve(
        "D-Log",
        0.9892,
        0.0108,
        0.256663,
        0.584555,
        6.025,
        0.0929,
        0.007827156200341792,
        0.1400586161070593,
        0.0078,
        0.14,
    ),
    _Curve(
        "F-Log",
        0.555556,
        0.009468,
        0.344676,
        0.790453,
        8.735631,
        0.092864,
        0.0005663467969879701,
        0.09781139663651882,
        0.00089,
        0.100537775223865,
    ),
    _Curve(
        "F-Log2",
        5.555556,
        0.064829,
        0.245281,
        0.384316,
        8.799461,
        0.092864,
        0.0008888881429483923,
        0.10068573654723681,
        0.000889,
        0.100686685370811,
    ),
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


def _encode(values: np.ndarray, curve: _Curve, *, printed: bool = False) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    cut = curve.printed_x if printed else curve.x
    result = np.empty_like(source)
    linear = source < cut
    result[linear] = curve.e * source[linear] + curve.f
    result[~linear] = curve.c * np.log10(curve.a * source[~linear] + curve.b) + curve.d
    return result


def _decode(values: np.ndarray, curve: _Curve, *, printed: bool = False) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    cut = curve.printed_encoded_cut if printed else curve.encoded_cut
    result = np.empty_like(source)
    linear = source < cut
    result[linear] = (source[linear] - curve.f) / curve.e
    result[~linear] = (10.0 ** ((source[~linear] - curve.d) / curve.c) - curve.b) / curve.a
    return result


def _panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=12, fill=_PANEL)
    draw.text((left + 16, top + 12), title, fill=_TEXT, font=ImageFont.load_default())
    plot = (left + 60, top + 44, right - 18, bottom - 32)
    x0, y0, x1, y1 = plot
    for fraction in np.linspace(0.0, 1.0, 5):
        x = round(x0 + fraction * (x1 - x0))
        y = round(y0 + fraction * (y1 - y0))
        draw.line((x, y0, x, y1), fill=_GRID)
        draw.line((x0, y, x1, y), fill=_GRID)
    return plot


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
    px_values = x0 + (np.asarray(x, dtype=np.float64) - xmin) * (x1 - x0) / (xmax - xmin)
    py_values = y1 - (np.asarray(y, dtype=np.float64) - ymin) * (y1 - y0) / (ymax - ymin)
    return [(round(px_value), round(py_value)) for px_value, py_value in zip(px_values, py_values, strict=True)]


def _sub_panels(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str
) -> tuple[tuple[int, int, int, int], ...]:
    """Draw one titled panel split into three side-by-side plots, one per curve."""
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=12, fill=_PANEL)
    draw.text((left + 16, top + 12), title, fill=_TEXT, font=ImageFont.load_default())
    gap = 12
    width = (right - left - 32 - 2 * gap) // 3
    plots = []
    for index, curve in enumerate(_CURVES):
        x0 = left + 16 + index * (width + gap)
        plot = (x0 + 10, top + 58, x0 + width - 10, bottom - 34)
        draw.text((x0 + 10, top + 38), curve.gamma, fill=_CURVE_COLORS[index], font=ImageFont.load_default())
        px0, py0, px1, py1 = plot
        for fraction in np.linspace(0.0, 1.0, 5):
            x = round(px0 + fraction * (px1 - px0))
            y = round(py0 + fraction * (py1 - py0))
            draw.line((x, py0, x, py1), fill=_GRID)
            draw.line((px0, y, px1, y), fill=_GRID)
        plots.append(plot)
    return tuple(plots)


def _vertical_marker(
    draw: ImageDraw.ImageDraw, plot: tuple[int, int, int, int], fraction: float, color: tuple[int, int, int]
) -> None:
    x0, y0, x1, y1 = plot
    x = round(x0 + fraction * (x1 - x0))
    for y in range(y0, y1, 6):
        draw.line((x, y, x, min(y + 3, y1)), fill=color)


def _legend(draw: ImageDraw.ImageDraw, origin: tuple[int, int]) -> None:
    x, y = origin
    for index, curve in enumerate(_CURVES):
        color = _CURVE_COLORS[index]
        draw.line((x, y + index * 18 + 6, x + 24, y + index * 18 + 6), fill=color, width=3)
        draw.text((x + 32, y + index * 18), curve.gamma, fill=_TEXT, font=ImageFont.load_default())


def _transfer_sheet() -> Image.Image:
    image = Image.new("RGB", (_WIDTH, 1340), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((24, 18), "Vendor A transfers — GPU float32 vs independent host float64", fill=_TEXT, font=font)
    draw.text(
        (24, 38),
        "Reflectance input; intersection cuts; negative/overshoot extension; no clipping or sign/magnitude mirror",
        fill=_MUTED,
        font=font,
    )

    full_encode = _panel(draw, (24, 72, 780, 390), "Signed full-domain encode (symlog reflectance axis)")
    full_decode = _panel(draw, (804, 72, 1576, 390), "Signed full-domain decode (symlog reflectance axis)")
    linear = np.concatenate((-np.geomspace(5e-5, 0.5, 512)[::-1], (0.0,), np.geomspace(5e-5, 64.0, 1536)))
    symlog = np.sign(linear) * np.log10(1.0 + 100.0 * np.abs(linear))
    encoded_axis = np.linspace(-0.5, 1.5, 2049)
    encode_values = np.concatenate(tuple(_encode(linear, curve) for curve in _CURVES))
    encode_range = (float(np.min(encode_values)), float(np.max(encode_values)))
    decode_values = np.concatenate(tuple(_decode(encoded_axis, curve) for curve in _CURVES))
    decode_display = np.sign(decode_values) * np.log10(1.0 + np.abs(decode_values))
    decode_range = (float(np.min(decode_display)), float(np.max(decode_display)))
    for index, curve in enumerate(_CURVES):
        color = _CURVE_COLORS[index]
        draw.line(
            _points(
                symlog,
                _encode(linear, curve),
                full_encode,
                x_range=(float(symlog[0]), float(symlog[-1])),
                y_range=encode_range,
            ),
            fill=color,
            width=2,
        )
        decoded = _decode(encoded_axis, curve)
        decoded_symlog = np.sign(decoded) * np.log10(1.0 + np.abs(decoded))
        draw.line(_points(encoded_axis, decoded_symlog, full_decode, y_range=decode_range), fill=color, width=2)
    _legend(draw, (92, 96))

    encode_zoom = _sub_panels(draw, (24, 414, 780, 748), "Encode cut enlargement per curve — GPU dots over oracle")
    decode_zoom = _sub_panels(draw, (804, 414, 1576, 748), "Decode cut enlargement per curve — GPU dots over oracle")
    for index, curve in enumerate(_CURVES):
        color = _CURVE_COLORS[index]
        half_width = max(curve.x * 0.08, 2e-6)
        x_values = np.linspace(curve.x - half_width, curve.x + half_width, 513)
        y_values = _encode(x_values, curve)
        plot = encode_zoom[index]
        x_range = (float(x_values[0]), float(x_values[-1]))
        y_pad = max(float(np.ptp(y_values)) * 0.05, 1e-12)
        y_range = (float(np.min(y_values)) - y_pad, float(np.max(y_values)) + y_pad)
        _vertical_marker(draw, plot, 0.5, _MUTED)
        draw.line(_points(x_values, y_values, plot, x_range=x_range, y_range=y_range), fill=color, width=2)
        gpu = _gpu_encode(x_values.astype(np.float32), curve.gamma)
        for point in _points(x_values[::24], gpu[::24], plot, x_range=x_range, y_range=y_range):
            draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=_GPU)
        draw.text(
            (plot[0], plot[3] + 6),
            f"X = {curve.x:.6g}  window +/-{half_width:.2g}",
            fill=_MUTED,
            font=ImageFont.load_default(),
        )

        half_encoded = max(curve.encoded_cut * 0.002, 3e-6)
        encoded_values = np.linspace(curve.encoded_cut - half_encoded, curve.encoded_cut + half_encoded, 513)
        decoded_values = _decode(encoded_values, curve)
        plot = decode_zoom[index]
        x_range = (float(encoded_values[0]), float(encoded_values[-1]))
        y_pad = max(float(np.ptp(decoded_values)) * 0.05, 1e-12)
        y_range = (float(np.min(decoded_values)) - y_pad, float(np.max(decoded_values)) + y_pad)
        _vertical_marker(draw, plot, 0.5, _MUTED)
        draw.line(_points(encoded_values, decoded_values, plot, x_range=x_range, y_range=y_range), fill=color, width=2)
        gpu_decoded = _gpu_decode(encoded_values.astype(np.float32), curve.gamma)
        for point in _points(encoded_values[::24], gpu_decoded[::24], plot, x_range=x_range, y_range=y_range):
            draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=_GPU)
        draw.text(
            (plot[0], plot[3] + 6),
            f"ENC_CUT = {curve.encoded_cut:.6g}  window +/-{half_encoded:.2g}",
            fill=_MUTED,
            font=ImageFont.load_default(),
        )

    difference = _sub_panels(
        draw, (24, 772, 780, 1100), "Intersection definition minus printed-cut definition (encode)"
    )
    residual = _panel(draw, (804, 772, 1576, 1100), "Round-trip residual across signed domain")
    for index, curve in enumerate(_CURVES):
        color = _CURVE_COLORS[index]
        lo, hi = sorted((curve.x, curve.printed_x))
        pad = max(abs(hi - lo), 1e-7)
        difference_x = np.linspace(lo - pad, hi + pad, 1025)
        delta = _encode(difference_x, curve) - _encode(difference_x, curve, printed=True)
        plot = difference[index]
        x_range = (float(difference_x[0]), float(difference_x[-1]))
        peak = max(float(np.max(np.abs(delta))), 1e-12)
        y_range = (-peak * 1.1, peak * 1.1)
        _vertical_marker(draw, plot, (curve.x - x_range[0]) / (x_range[1] - x_range[0]), color)
        _vertical_marker(draw, plot, (curve.printed_x - x_range[0]) / (x_range[1] - x_range[0]), _MUTED)
        draw.line(_points(difference_x, delta, plot, x_range=x_range, y_range=y_range), fill=color, width=2)
        draw.text(
            (plot[0], plot[3] + 6),
            f"max |delta| = {peak:.2g}  (coloured: X, grey: printed cut)",
            fill=_MUTED,
            font=ImageFont.load_default(),
        )

        roundtrip_x = np.linspace(-0.5, 64.0, 4097, dtype=np.float64).astype(np.float32)
        roundtrip = _gpu_decode(_gpu_encode(roundtrip_x, curve.gamma), curve.gamma)
        error = roundtrip.astype(np.float64) - roundtrip_x.astype(np.float64)
        draw.line(
            _points(roundtrip_x, error, residual, x_range=(-0.5, 64.0), y_range=(-2e-4, 2e-4)), fill=color, width=2
        )

    anchor_y = 1130
    draw.text((24, anchor_y), "Published anchors: GPU y*1023 / independent oracle y*1023", fill=_TEXT, font=font)
    for index, curve in enumerate(_CURVES):
        anchors = np.asarray((0.0, 0.18, 0.9), dtype=np.float32)
        gpu_codes = _gpu_encode(anchors, curve.gamma).astype(np.float64) * 1023.0
        oracle_codes = _encode(anchors.astype(np.float64), curve) * 1023.0
        text = f"{curve.gamma}: GPU {gpu_codes[0]:.4f} / {gpu_codes[1]:.4f} / {gpu_codes[2]:.4f}    oracle {oracle_codes[0]:.4f} / {oracle_codes[1]:.4f} / {oracle_codes[2]:.4f}"
        draw.text((24, anchor_y + 24 + index * 24), text, fill=_CURVE_COLORS[index], font=font)
    draw.text(
        (24, 1245),
        "teal dots: GPU float32    coloured lines: independent float64    dashed: cut positions",
        fill=_MUTED,
        font=font,
    )
    return image


def _display(values: np.ndarray) -> np.ndarray:
    frame = _frame(values.reshape(-1, 3), colorspace="Rec.709", gamma="linear")
    encoded = px.color.linear_to_gamma(frame, gamma="sRGB")
    array = px.io.to_array(encoded).get()[0].reshape(values.shape)
    return np.asarray(np.clip(array, 0.0, 1.0) * 255.0 + 0.5, dtype=np.uint8)


def _gamut_sheet() -> Image.Image:
    image = Image.new("RGB", (_WIDTH, 820), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text(
        (24, 18), "Vendor A gamuts — representative coordinate-derived conversion to Rec.709", fill=_TEXT, font=font
    )
    draw.text(
        (24, 40),
        "Production: published primaries + D65; D65 identity; Bradford only for differing whites",
        fill=_MUTED,
        font=font,
    )
    width, height = 720, 260
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    source = np.stack(
        (np.broadcast_to(x, (height, width)), np.broadcast_to(y, (height, width)), 1.2 - 0.8 * x - 0.5 * y), axis=-1
    )
    definitions = (
        (
            "D-Gamut",
            ((0.71, 0.31), (0.21, 0.88), (0.09, -0.08)),
            (0.283004662361243, 0.813196056391736, -0.096200718752979),
        ),
        (
            "F-Gamut-C",
            ((0.7347, 0.2653), (0.0263, 0.9737), (0.1173, -0.0224)),
            (0.285007008240737, 0.741945697114496, -0.026952705355233),
        ),
    )
    for index, (colorspace, primaries, native_row) in enumerate(definitions):
        frame = _frame(source.reshape(-1, 3), colorspace=colorspace)
        converted = px.color.rgb_to_rgb(frame, output_colorspace="Rec.709", output_gamma="linear")
        rec709 = px.io.to_array(converted).get()[0].reshape(source.shape)
        strip = Image.fromarray(_display(rec709), mode="RGB")
        top = 88 + index * 344
        image.paste(strip, (24, top))
        draw.text(
            (24, top - 18),
            f"{colorspace} source coordinates interpreted and converted to Rec.709",
            fill=_CURVE_COLORS[index],
            font=font,
        )
        draw.text(
            (770, top + 18), f"Primaries: R {primaries[0]}  G {primaries[1]}  B {primaries[2]}", fill=_TEXT, font=font
        )
        draw.text((770, top + 44), "White: D65 (0.3127, 0.3290)", fill=_TEXT, font=font)
        draw.text((770, top + 70), f"native row: {native_row}", fill=_TEXT, font=font)
        draw.text((770, top + 104), "Auxiliary matrices: vendor / CAT02 checks only", fill=_MUTED, font=font)
        draw.text((770, top + 128), "Production differing-white adaptation: Bradford", fill=_MUTED, font=font)
        swatch_values = np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.18, 0.18, 0.18)), np.float32)
        swatch_frame = _frame(swatch_values, colorspace=colorspace)
        swatch_rec709 = px.io.to_array(px.color.rgb_to_rgb(swatch_frame, output_colorspace="Rec.709")).get()[0]
        swatches = _display(swatch_rec709.reshape(1, 4, 3))[0]
        for swatch_index, color in enumerate(swatches):
            left = 770 + swatch_index * 120
            draw.rectangle((left, top + 172, left + 96, top + 230), fill=tuple(int(channel) for channel in color))
    return image


def generate(directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    transfer_path = directory / "vendor-a-transfers.png"
    gamut_path = directory / "vendor-a-gamuts.png"
    _transfer_sheet().save(transfer_path, format="PNG", optimize=False, compress_level=9)
    _gamut_sheet().save(gamut_path, format="PNG", optimize=False, compress_level=9)
    return transfer_path, gamut_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    for path in generate(args.directory):
        print(path)


if __name__ == "__main__":
    main()
