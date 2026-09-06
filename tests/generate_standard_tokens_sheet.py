"""Generate deterministic visual evidence for v1-standard-tokens acceptance 138."""

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
_ACES_WHITE = (0.32168, 0.33767)
_P3_PRIMARIES = ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060))
_DEFINITIONS = {
    "P3-DCI": (_P3_PRIMARIES, (0.3140, 0.3510)),
    "P3-D60": (_P3_PRIMARIES, _ACES_WHITE),
    "P3-D65": (_P3_PRIMARIES, _D65),
    "SMPTE-C": (((0.630, 0.340), (0.310, 0.595), (0.155, 0.070)), _D65),
    "Rec.709": (((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)), _D65),
}
_BRADFORD = np.asarray(
    ((0.8951, 0.2664, -0.1614), (-0.7502, 1.7135, 0.0367), (0.0389, -0.0685, 1.0296)),
    dtype=np.float64,
)


def _frame(values: np.ndarray, *, colorspace: str = "ACEScg", gamma: str = "linear") -> px.core.Frame:
    rgb = np.repeat(np.asarray(values, dtype=np.float32)[:, None], 3, axis=1)[None]
    return px.io.from_array(cp.asarray(rgb), colorspace=colorspace, gamma=gamma, channels="RGB")


def _encode(values: np.ndarray, gamma: str) -> np.ndarray:
    encoded = px.color.linear_to_gamma(_frame(values), gamma=gamma)
    return px.io.to_array(encoded).get()[0, :, 0].astype(np.float64)


def _decode(values: np.ndarray, gamma: str) -> np.ndarray:
    decoded = px.color.gamma_to_linear(_frame(values, gamma=gamma), gamma=gamma)
    return px.io.to_array(decoded).get()[0, :, 0].astype(np.float64)


def _acescc_encode(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    result = np.empty_like(x)
    nonpositive = x <= 0.0
    lower = (x > 0.0) & (x < 2.0**-15)
    upper = ~(nonpositive | lower)
    result[nonpositive] = (-16.0 + 9.72) / 17.52
    result[lower] = (np.log2(2.0**-16 + x[lower] / 2.0) + 9.72) / 17.52
    result[upper] = (np.log2(x[upper]) + 9.72) / 17.52
    return result


def _acescct_encode(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    linear = x <= 0.0078125
    result = np.empty_like(x)
    result[linear] = 10.5402377416545 * x[linear] + 0.0729055341958355
    result[~linear] = (np.log2(x[~linear]) + 9.72) / 17.52
    return result


def _acescc_decode(values: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64)
    lower = y <= (9.72 - 15.0) / 17.52
    result = np.empty_like(y)
    result[lower] = 2.0 * (2.0 ** (17.52 * y[lower] - 9.72) - 2.0**-16)
    result[~lower] = 2.0 ** (17.52 * y[~lower] - 9.72)
    return result


def _acescct_decode(values: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64)
    linear = y <= 0.155251141552511
    result = np.empty_like(y)
    result[linear] = (y[linear] - 0.0729055341958355) / 10.5402377416545
    result[~linear] = 2.0 ** (17.52 * y[~linear] - 9.72)
    return result


def _map(values: np.ndarray, lower: float, upper: float, start: int, extent: int) -> np.ndarray:
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
    lower = float(finite.min())
    upper = float(finite.max())
    padding = max((upper - lower) * 0.05, 1e-12)
    lower -= padding
    upper += padding
    draw.rectangle((left, top, left + width, top + height), outline=_GRID)
    draw.text((left, top - 21), title, fill=_TEXT, font=font)
    xp = _map(np.asarray(x, dtype=np.float64), float(x[0]), float(x[-1]), left, width)
    for values, color in curves:
        yp = top + height - _map(np.asarray(values), lower, upper, 0, height)
        draw.line(tuple(zip(xp, yp, strict=True)), fill=color, width=2)
    draw.text((left, top + height + 4), f"x {float(x[0]):.7g} .. {float(x[-1]):.7g}", fill=_TEXT, font=font)
    draw.text((left + width - 230, top + height + 4), f"y {lower:.7g} .. {upper:.7g}", fill=_TEXT, font=font)


def _transfer_sheet() -> Image.Image:
    image = Image.new("RGB", (_WIDTH, 1880), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width = (_WIDTH - _LEFT - _RIGHT - 30) // 2
    draw.text((_LEFT, 18), "Standard transfer tokens: GPU float32 vs independent host float64", fill=_TEXT, font=font)
    draw.text((_LEFT, 40), "GPU", fill=_GPU, font=font)
    draw.text((_LEFT + 50, 40), "oracle", fill=_ORACLE, font=font)

    signed = np.linspace(-0.25, 2.0, _SAMPLES, dtype=np.float64).astype(np.float32)
    _panel(
        draw,
        font,
        box=(_LEFT, 90, width, 230),
        title="ACEScc signed encode (nonpositive collapse)",
        x=signed,
        curves=((_acescc_encode(signed), _ORACLE), (_encode(signed, "ACEScc"), _GPU)),
    )
    _panel(
        draw,
        font,
        box=(_LEFT + width + 30, 90, width, 230),
        title="ACEScct signed encode (linear negative toe)",
        x=signed,
        curves=((_acescct_encode(signed), _ORACLE), (_encode(signed, "ACEScct"), _GPU)),
    )

    near_zero = np.linspace(-(2.0**-18), 2.0**-14, _SAMPLES, dtype=np.float64).astype(np.float32)
    _panel(
        draw,
        font,
        box=(_LEFT, 390, width, 230),
        title="ACEScc cut zoom: 0 and 2^-15",
        x=near_zero,
        curves=((_acescc_encode(near_zero), _ORACLE), (_encode(near_zero, "ACEScc"), _GPU)),
    )
    near_cct = np.linspace(0.00780, 0.007825, _SAMPLES, dtype=np.float64).astype(np.float32)
    _panel(
        draw,
        font,
        box=(_LEFT + width + 30, 390, width, 230),
        title="ACEScct encode cut zoom: x=0.0078125",
        x=near_cct,
        curves=((_acescct_encode(near_cct), _ORACLE), (_encode(near_cct, "ACEScct"), _GPU)),
    )

    cc_cut = (9.72 - 15.0) / 17.52
    encoded_cc = np.linspace(cc_cut - 2e-5, cc_cut + 2e-5, _SAMPLES, dtype=np.float64).astype(np.float32)
    _panel(
        draw,
        font,
        box=(_LEFT, 690, width, 230),
        title="ACEScc decode threshold zoom",
        x=encoded_cc,
        curves=((_acescc_decode(encoded_cc), _ORACLE), (_decode(encoded_cc, "ACEScc"), _GPU)),
    )
    encoded_cct = np.linspace(0.155231, 0.155271, _SAMPLES, dtype=np.float64).astype(np.float32)
    _panel(
        draw,
        font,
        box=(_LEFT + width + 30, 690, width, 230),
        title="ACEScct decode threshold zoom (<=1 ULP reversal)",
        x=encoded_cct,
        curves=((_acescct_decode(encoded_cct), _ORACLE), (_decode(encoded_cct, "ACEScct"), _GPU)),
    )

    positive = np.linspace(0.0, 1.5, _SAMPLES, dtype=np.float64).astype(np.float32)
    powers = tuple(
        (_encode(positive, gamma), color)
        for gamma, color in (("Gamma-2.2", _ACCENT), ("Gamma-2.4", _ORACLE), ("Gamma-2.5", _GPU), ("Gamma-2.6", _ERROR))
    )
    _panel(
        draw,
        font,
        box=(_LEFT, 990, width, 230),
        title="Power comparison: 2.2 / 2.4 / 2.5 / 2.6",
        x=positive,
        curves=powers,
    )
    restored_cc = _decode(_encode(signed, "ACEScc"), "ACEScc")
    restored_cct = _decode(_encode(signed, "ACEScct"), "ACEScct")
    residual_cc = np.abs(restored_cc - signed.astype(np.float64))
    residual_cc[signed <= 0.0] = 0.0
    _panel(
        draw,
        font,
        box=(_LEFT + width + 30, 990, width, 230),
        title="Round-trip residual (ACEScc injective domain / ACEScct)",
        x=signed,
        curves=((residual_cc, _ORACLE), (np.abs(restored_cct - signed.astype(np.float64)), _GPU)),
    )

    draw.text(
        (_LEFT, 1300), "Published anchors: linear x | ACEScc GPU/oracle | ACEScct GPU/oracle", fill=_TEXT, font=font
    )
    anchors = np.asarray((0.0, 2.0**-15, 0.0078125, 0.18, 1.0, 65504.0), dtype=np.float32)
    cc_gpu = _encode(anchors, "ACEScc")
    cct_gpu = _encode(anchors, "ACEScct")
    cc_oracle = _acescc_encode(anchors)
    cct_oracle = _acescct_encode(anchors)
    for row, values in enumerate(zip(anchors, cc_gpu, cc_oracle, cct_gpu, cct_oracle, strict=True)):
        x, ccg, cco, cctg, ccto = values
        draw.text(
            (_LEFT, 1325 + row * 24),
            f"{float(x):>12.7g} | {ccg:>13.9f} / {cco:>13.9f} | {cctg:>13.9f} / {ccto:>13.9f}",
            fill=_TEXT,
            font=font,
        )
    draw.text(
        (_LEFT, 1490),
        f"max encode |GPU-oracle|: ACEScc={np.max(np.abs(cc_gpu - cc_oracle)):.3e}; "
        f"ACEScct={np.max(np.abs(cct_gpu - cct_oracle)):.3e}",
        fill=_TEXT,
        font=font,
    )
    draw.text(
        (_LEFT, 1514),
        f"max round-trip residual: ACEScc positive={residual_cc[signed > 0].max():.3e}; "
        f"ACEScct signed={np.max(np.abs(restored_cct - signed)):.3e}",
        fill=_TEXT,
        font=font,
    )
    draw.text(
        (_LEFT, 1548),
        "ACEScc nonpositive collapse and ACEScct public-decimal cut residual are defined behavior.",
        fill=_TEXT,
        font=font,
    )
    draw.text(
        (_LEFT, 1572),
        "No 65504 upper clip is applied; preview panels do not redefine numeric values.",
        fill=_TEXT,
        font=font,
    )
    return image


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


def _display_strip(values: np.ndarray, height: int) -> np.ndarray:
    preview = np.power(np.clip(values, 0.0, 1.0), np.float32(1.0 / 2.2))
    row = np.rint(preview * np.float32(255.0)).astype(np.uint8)
    return np.repeat(row[None, :, :], height, axis=0)


def _gamut_sheet() -> Image.Image:
    image = Image.new("RGB", (_WIDTH, 1290), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    strip_width = _WIDTH - _LEFT - _RIGHT
    x = np.linspace(0.0, 1.0, strip_width, dtype=np.float32)
    source_values = np.stack((1.35 * x - 0.15, 0.18 + 0.82 * np.sin(np.pi * x), 1.2 * (1.0 - x) - 0.1), axis=1)
    draw.text((_LEFT, 18), "P3 variants and SMPTE-C to Rec.709 / linear", fill=_TEXT, font=font)
    draw.text(
        (_LEFT, 40),
        "Scene values include negative components and overshoot; only the display preview clips.",
        fill=_TEXT,
        font=font,
    )
    for row, token in enumerate(("P3-DCI", "P3-D60", "P3-D65", "SMPTE-C")):
        source = px.io.from_array(cp.asarray(source_values[None]), colorspace=token, gamma="linear", channels="RGB")
        actual = px.io.to_array(px.color.rgb_to_rgb(source, output_colorspace="Rec.709", output_gamma="linear")).get()[
            0
        ]
        oracle = source_values.astype(np.float64) @ _conversion(token, "Rec.709").T
        top = 92 + row * 285
        image.paste(Image.fromarray(_display_strip(actual, 180), mode="RGB"), (_LEFT, top))
        draw.rectangle((_LEFT, top, _LEFT + strip_width - 1, top + 179), outline=_GRID)
        draw.text((28, top + 78), token, fill=_TEXT, font=font)
        draw.text(
            (_LEFT, top + 190),
            f"white={_DEFINITIONS[token][1]!r}; min={actual.min():.7f}; max={actual.max():.7f}; "
            f"max |GPU-oracle|={np.max(np.abs(actual - oracle)):.3e}",
            fill=_TEXT,
            font=font,
        )
    return image


def generate(directory: Path) -> tuple[Path, Path]:
    """Generate deterministic transfer and gamut comparison images."""
    directory.mkdir(parents=True, exist_ok=True)
    transfer = directory / "standard-transfer-curves.png"
    gamut = directory / "standard-gamut-conversions.png"
    _transfer_sheet().save(transfer, format="PNG", compress_level=9)
    _gamut_sheet().save(gamut, format="PNG", compress_level=9)
    return transfer, gamut


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(".nf/tmp/sheets-17-standard"))
    arguments = parser.parse_args()
    for path in generate(arguments.output_dir):
        print(path)


if __name__ == "__main__":
    main()
