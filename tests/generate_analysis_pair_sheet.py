"""Generate the manual visual-acceptance sheet for v1-analysis-pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import cupy as cp
import numpy as np

import pixtreme as px

_PANEL_WIDTH = 220
_PANEL_HEIGHT = 140
_LABEL_HEIGHT = 38
_METHODS = ("sqdiff", "sqdiff_normed", "ccorr", "ccorr_normed", "ccoeff", "ccoeff_normed")


def _response_frame(response: cp.ndarray) -> px.core.Frame:
    """Cross the raw response boundary explicitly for sheet-only visualization."""
    return px.io.from_array(
        response[..., None],
        colorspace="ACEScg",
        gamma="linear",
        channels=["response"],
    )


def _remap_response(response: cp.ndarray) -> tuple[px.core.Frame, float, float]:
    raw = _response_frame(response)
    raw_min = float(cp.min(raw.data))
    raw_max = float(cp.max(raw.data))
    finite = cp.where(cp.isfinite(raw.data), raw.data, np.float32(0.0))
    finite_min = cp.min(finite)
    finite_max = cp.max(finite)
    span = finite_max - finite_min
    remapped = cp.where(span > np.float32(0.0), (finite - finite_min) / span, np.float32(0.0))
    rgb = cp.repeat(remapped, 3, axis=2)
    display = px.io.from_array(rgb, colorspace="sRGB", gamma="sRGB", channels="RGB")
    return px.transform.resize(display, width=_PANEL_WIDTH, height=_PANEL_HEIGHT), raw_min, raw_max


def _display_source(frame: px.core.Frame) -> tuple[px.core.Frame, float, float]:
    raw_min = float(cp.min(frame.data))
    raw_max = float(cp.max(frame.data))
    span = np.float32(raw_max - raw_min)
    if span > np.float32(0.0):
        mapped = (frame.data - np.float32(raw_min)) / span
    else:
        mapped = cp.zeros_like(frame.data)
    if mapped.shape[2] == 1:
        mapped = cp.repeat(mapped, 3, axis=2)
    display = px.io.from_array(mapped, colorspace="sRGB", gamma="sRGB", channels="RGB")
    return px.transform.resize(display, width=_PANEL_WIDTH, height=_PANEL_HEIGHT), raw_min, raw_max


def _label(display: px.core.Frame, text: str) -> px.core.Frame:
    label_data = cp.full((_LABEL_HEIGHT, _PANEL_WIDTH, 3), np.float32(0.015), dtype=cp.float32)
    bar = px.io.from_array(label_data, colorspace="sRGB", gamma="sRGB", channels="RGB")
    bar = px.draw.text(
        bar,
        text=text,
        position=(4.0, 12.0),
        size=8.0,
        color=(1.0, 1.0, 1.0),
        anchor="baseline-left",
        font="mono",
        line_spacing=0.9,
    )
    return px.transform.stack((bar, display), direction="vertical")


def _source_panel(frame: px.core.Frame, name: str) -> px.core.Frame:
    display, raw_min, raw_max = _display_source(frame)
    return _label(display, f"{name}\nraw=[{raw_min:.3g},{raw_max:.3g}]")


def _response_panel(response: cp.ndarray, name: str, *, direction: str | None = None) -> px.core.Frame:
    display, raw_min, raw_max = _remap_response(response)
    suffix = ""
    if direction is not None:
        index = int(cp.argmin(response).get()) if direction == "min" else int(cp.argmax(response).get())
        best_y, best_x = np.unravel_index(index, response.shape)
        suffix = f" best-{direction}=({best_y},{best_x})"
    return _label(display, f"{name}{suffix}\nraw=[{raw_min:.3g},{raw_max:.3g}]")


def _corner_source() -> px.core.Frame:
    y, x = cp.indices((_PANEL_HEIGHT, _PANEL_WIDTH), dtype=cp.float32)
    data = cp.full((_PANEL_HEIGHT, _PANEL_WIDTH, 3), np.float32(-0.35), dtype=cp.float32)
    data[..., 0] += cp.where((x >= 28) & (x < 104) & (y >= 26) & (y < 112), np.float32(1.8), np.float32(0.0))
    data[..., 1] += cp.where(x >= np.float32(122.0), (x - np.float32(122.0)) / np.float32(52.0), 0.0)
    data[..., 2] += cp.where(y >= np.float32(72.0), (y - np.float32(72.0)) / np.float32(34.0), 0.0)
    return px.io.from_array(data, colorspace="ACEScg", gamma="linear", channels=("A", "custom", "Z"))


def _frame_from_host(values: np.ndarray) -> px.core.Frame:
    return px.io.from_array(cp.asarray(values), colorspace="ACEScg", gamma="linear", channels="RGB")


def _channel_frame(frame: px.core.Frame, index: int, label: str) -> px.core.Frame:
    return px.io.from_array(
        frame.data[..., index : index + 1],
        colorspace=frame.colorspace,
        gamma=frame.gamma,
        channels=[label],
        matrix=frame.matrix,
    )


def _matching_material() -> tuple[px.core.Frame, px.core.Frame]:
    height, width = _PANEL_HEIGHT, _PANEL_WIDTH
    values = np.full((height, width, 3), -0.2, dtype=np.float32)
    yy, xx = np.indices((24, 24), dtype=np.float32)
    pattern = np.empty((24, 24, 3), dtype=np.float32)
    pattern[..., 0] = np.where((xx < 12) ^ (yy < 12), 1.5, -0.5)
    pattern[..., 1] = np.sin(xx * np.float32(0.45)) + np.float32(0.15) * yy
    pattern[..., 2] = np.where((xx - 12) ** 2 + (yy - 12) ** 2 < 8**2, 1.2, -0.4)
    values[31:55, 26:50, :] = pattern
    values[31:55, 78:102, :] = pattern + np.asarray([0.55, -0.35, 0.8], dtype=np.float32)
    values[31:55, 130:154, :] = pattern * np.asarray([1.35, 0.65, 1.1], dtype=np.float32)
    values[86:110, 170:194, :] = pattern[..., (2, 1, 0)]
    return _frame_from_host(values), _frame_from_host(pattern)


def generate_sheet(path: Path) -> None:
    corner_source = _corner_source()
    corner_row = px.transform.stack(
        (
            _source_panel(corner_source, "HARRIS INPUT"),
            _response_panel(px.feature.corner_harris(corner_source), "block=3 k=.04 mirror"),
            _response_panel(px.feature.corner_harris(corner_source, block_size=1), "block=1 k=.04 mirror"),
            _response_panel(px.feature.corner_harris(corner_source, block_size=7), "block=7 k=.04 mirror"),
            _response_panel(px.feature.corner_harris(corner_source, k=0.02), "block=3 k=.02 mirror"),
            _response_panel(px.feature.corner_harris(corner_source, k=0.12), "block=3 k=.12 mirror"),
        ),
        direction="horizontal",
    )
    border_row = px.transform.stack(
        (
            _response_panel(px.feature.corner_harris(corner_source, border="mirror"), "BORDER mirror"),
            _response_panel(px.feature.corner_harris(corner_source, border="replicate"), "BORDER replicate"),
            _response_panel(px.feature.corner_harris(corner_source, border="wrap"), "BORDER wrap"),
            _response_panel(
                px.feature.corner_harris(corner_source, border="constant", border_value=-0.7),
                "BORDER constant=-.7",
            ),
            _source_panel(corner_source, "flat / edge / corner"),
            _source_panel(corner_source, "orthogonal channels"),
        ),
        direction="horizontal",
    )
    match_source, template = _matching_material()
    material_row = px.transform.stack(
        (
            _source_panel(match_source, "MATCH SEARCH IMAGE"),
            _source_panel(template, "MATCH TEMPLATE"),
            _source_panel(_channel_frame(match_source, 0, "R"), "SEARCH channel R"),
            _source_panel(_channel_frame(match_source, 1, "G"), "SEARCH channel G"),
            _source_panel(_channel_frame(match_source, 2, "B"), "SEARCH channel B"),
            _source_panel(_channel_frame(template, 1, "G"), "TEMPLATE channel G"),
        ),
        direction="horizontal",
    )
    match_panels = []
    for method in _METHODS:
        response = px.feature.match_template(match_source, template, method=method)
        direction = "min" if method.startswith("sqdiff") else "max"
        match_panels.append(_response_panel(response, method, direction=direction))
    match_row = px.transform.stack(tuple(match_panels), direction="horizontal")
    sheet = px.transform.stack((corner_row, border_row, material_row, match_row), direction="vertical")
    code = cp.rint(cp.clip(sheet.data, 0.0, 1.0) * np.float32(255.0)).astype(cp.uint8)
    output = px.io.from_array(code, colorspace="sRGB", gamma="sRGB", channels="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    px.io.write_image(path, output, compression_level=6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="PNG output path")
    arguments = parser.parse_args()
    generate_sheet(arguments.output)


if __name__ == "__main__":
    main()
