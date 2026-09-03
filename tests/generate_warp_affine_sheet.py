"""Generate the manual visual-acceptance sheet for v1-warp-affine."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cupy as cp
import numpy as np

import pixtreme as px

_SOURCE_WIDTH = 360
_SOURCE_HEIGHT = 220
_PANEL_WIDTH = 180
_PANEL_HEIGHT = 110
_LABEL_HEIGHT = 40
_INTERPOLATIONS = (
    "nearest",
    "bilinear",
    "bicubic",
    "b-spline",
    "mitchell",
    "lanczos2",
    "lanczos3",
    "lanczos4",
    "area",
)


def _centered_matrix(
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    rotation: float = 0.0,
    shear: float = 0.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> np.ndarray:
    angle = math.radians(rotation)
    rotation_matrix = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float32,
    )
    scale_shear = np.asarray([[scale_x, shear], [0.0, scale_y]], dtype=np.float32)
    linear = rotation_matrix @ scale_shear
    center = np.asarray(((_SOURCE_WIDTH - 1) / 2.0, (_SOURCE_HEIGHT - 1) / 2.0), dtype=np.float32)
    translation = center - linear @ center + np.asarray((offset_x, offset_y), dtype=np.float32)
    return np.column_stack((linear, translation)).astype(np.float32)


def _diagnostic_source() -> px.core.Frame:
    y, x = cp.indices((_SOURCE_HEIGHT, _SOURCE_WIDTH), dtype=cp.float32)
    checker = cp.mod(cp.floor(x / np.float32(7.0)) + cp.floor(y / np.float32(7.0)), np.float32(2.0))
    diagonal = cp.where(cp.abs(y - np.float32(0.47) * x - np.float32(17.0)) < np.float32(0.8), 1.0, 0.0)
    fine_vertical = cp.where(cp.mod(x, np.float32(13.0)) < np.float32(1.0), 1.0, 0.0)
    radial = cp.sin(cp.hypot(x - np.float32(270.0), y - np.float32(72.0)) * np.float32(0.42))
    data = cp.empty((_SOURCE_HEIGHT, _SOURCE_WIDTH, 3), dtype=cp.float32)
    data[..., 0] = np.float32(-0.25) + np.float32(1.35) * checker
    data[..., 1] = np.float32(0.12) + np.float32(1.25) * diagonal + np.float32(0.25) * radial
    data[..., 2] = np.float32(-0.10) + np.float32(1.20) * fine_vertical
    frame = px.io.from_array(
        data,
        colorspace="ACEScg",
        gamma="linear",
        channels=("temperature", "mask", "Z"),
    )
    return px.draw.text(
        frame,
        text="AFFINE 0.5px / RGB? NO: T M Z",
        position=(16.0, 194.0),
        size=18.0,
        color=(1.6, -0.2, 1.2),
        anchor="baseline-left",
        font="mono",
    )


def _photo_like_source() -> px.core.Frame:
    y, x = cp.indices((_SOURCE_HEIGHT, _SOURCE_WIDTH), dtype=cp.float32)
    normalized_x = x / np.float32(_SOURCE_WIDTH - 1)
    normalized_y = y / np.float32(_SOURCE_HEIGHT - 1)
    horizon = np.float32(0.57) + np.float32(0.035) * cp.sin(normalized_x * np.float32(17.0))
    sky = cp.clip((horizon - normalized_y) * np.float32(3.7), 0.0, 1.0)
    sun = cp.exp(
        -(
            (normalized_x - np.float32(0.72)) ** 2 / np.float32(0.004)
            + (normalized_y - np.float32(0.22)) ** 2 / np.float32(0.012)
        )
    )
    ridge = cp.where(
        normalized_y
        > np.float32(0.50)
        + np.float32(0.08) * cp.sin(normalized_x * np.float32(12.0))
        + np.float32(0.025) * cp.sin(normalized_x * np.float32(43.0)),
        1.0,
        0.0,
    )
    texture = cp.sin(x * np.float32(0.71) + y * np.float32(1.13)) * cp.sin(x * np.float32(1.91) - y * np.float32(0.37))
    data = cp.empty((_SOURCE_HEIGHT, _SOURCE_WIDTH, 3), dtype=cp.float32)
    data[..., 0] = np.float32(0.08) + np.float32(0.44) * sky + np.float32(1.25) * sun
    data[..., 1] = np.float32(0.10) + np.float32(0.66) * sky + np.float32(0.20) * ridge
    data[..., 2] = np.float32(0.12) + np.float32(0.98) * sky - np.float32(0.04) * ridge
    data[..., 0] += ridge * texture * np.float32(0.07)
    data[..., 1] += ridge * texture * np.float32(0.12)
    data[..., 2] += ridge * texture * np.float32(0.05)
    return px.io.from_array(data, colorspace="sRGB", gamma="sRGB", channels="RGB")


def _display(frame: px.core.Frame) -> px.core.Frame:
    rgb = frame.data[..., :3]
    display = cp.rint(cp.clip(rgb, 0.0, 1.0) * np.float32(255.0)).astype(cp.uint8)
    return px.io.from_array(display, colorspace="sRGB", gamma="sRGB", channels="RGB")


def _panel(frame: px.core.Frame, label: str) -> px.core.Frame:
    minimum = float(cp.min(frame.data).get())
    maximum = float(cp.max(frame.data).get())
    image = px.transform.resize(_display(frame), width=_PANEL_WIDTH, height=_PANEL_HEIGHT, interpolation="bilinear")
    label_data = cp.full((_LABEL_HEIGHT, _PANEL_WIDTH, 3), np.uint8(4), dtype=cp.uint8)
    bar = px.io.from_array(label_data, colorspace="sRGB", gamma="sRGB", channels="RGB")
    bar = px.values.cast_dtype(bar, dtype="float32")
    bar = px.draw.text(
        bar,
        text=f"{label}\nraw {minimum:+.2f}..{maximum:+.2f}",
        position=(4.0, 12.0),
        size=8.0,
        color=(255.0, 255.0, 255.0),
        anchor="baseline-left",
        font="mono",
        line_spacing=0.9,
    )
    return px.transform.stack((bar, image), direction="vertical")


def _warp(
    frame: px.core.Frame,
    matrix: np.ndarray,
    *,
    interpolation: str | None = None,
    inverse: bool = False,
    border: str = "constant",
    border_value: float | None = None,
) -> px.core.Frame:
    return px.transform.warp_affine(
        frame,
        matrix,
        inverse=inverse,
        interpolation=interpolation,
        border=border,
        border_value=border_value,
    )


def generate_sheet(path: Path) -> None:
    diagnostic = _diagnostic_source()
    photo = _photo_like_source()
    translation = _centered_matrix(offset_x=48.0, offset_y=-24.0)
    rotation = _centered_matrix(rotation=17.0)
    shear = _centered_matrix(shear=0.38)
    enlarge = _centered_matrix(scale_x=1.35, scale_y=1.35)
    shrink = _centered_matrix(scale_x=0.68, scale_y=0.68)
    mixed = _centered_matrix(scale_x=1.28, scale_y=0.62, rotation=-9.0, shear=0.22)
    forward = _warp(diagnostic, translation, interpolation="nearest")
    restored = _warp(forward, translation, inverse=True, interpolation="nearest")
    geometry_row = px.transform.stack(
        (
            _panel(photo, "PHOTO-LIKE INPUT"),
            _panel(diagnostic, "DIAGNOSTIC INPUT"),
            _panel(_warp(photo, translation), "TRANSLATE"),
            _panel(_warp(photo, rotation), "ROTATE 17deg"),
            _panel(_warp(photo, shear), "SHEAR .38"),
            _panel(_warp(photo, enlarge), "SCALE 1.35"),
            _panel(_warp(photo, shrink), "SCALE .68"),
            _panel(_warp(photo, mixed), "MIXED SCALE"),
            _panel(forward, "FORWARD nearest"),
            _panel(restored, "INVERSE RETURN"),
        ),
        direction="horizontal",
    )

    interpolation_matrix = _centered_matrix(scale_x=0.74, scale_y=0.81, rotation=11.0, shear=0.16)
    interpolation_row = px.transform.stack(
        (
            _panel(diagnostic, "9 KERNEL INPUT"),
            *(
                _panel(
                    _warp(diagnostic, interpolation_matrix, interpolation=interpolation, border="mirror"),
                    interpolation,
                )
                for interpolation in _INTERPOLATIONS
            ),
        ),
        direction="horizontal",
    )

    auto_area = _centered_matrix(scale_x=0.78, scale_y=1.12, rotation=7.0)
    auto_lanczos = _centered_matrix(scale_x=1.05, scale_y=1.08, rotation=5.0)
    border_matrix = _centered_matrix(rotation=13.0, offset_x=62.0, offset_y=-31.0)
    border_row = px.transform.stack(
        (
            _panel(_warp(diagnostic, auto_area), "AUTO area"),
            _panel(_warp(diagnostic, auto_area, interpolation="area"), "EXPLICIT area"),
            _panel(_warp(diagnostic, auto_lanczos), "AUTO lanczos4"),
            _panel(_warp(diagnostic, auto_lanczos, interpolation="lanczos4"), "EXPLICIT lanczos4"),
            _panel(_warp(diagnostic, border_matrix, interpolation="bilinear", border="mirror"), "BORDER mirror"),
            _panel(
                _warp(diagnostic, border_matrix, interpolation="bilinear", border="replicate"),
                "BORDER replicate",
            ),
            _panel(_warp(diagnostic, border_matrix, interpolation="bilinear", border="wrap"), "BORDER wrap"),
            _panel(_warp(diagnostic, border_matrix, interpolation="bilinear"), "CONSTANT default 0"),
            _panel(
                _warp(diagnostic, border_matrix, interpolation="bilinear", border_value=-0.5),
                "CONSTANT -0.5",
            ),
            _panel(
                _warp(diagnostic, border_matrix, interpolation="bilinear", border_value=1.5),
                "CONSTANT +1.5",
            ),
        ),
        direction="horizontal",
    )

    sheet = px.transform.stack((geometry_row, interpolation_row, border_row), direction="vertical")
    path.parent.mkdir(parents=True, exist_ok=True)
    px.io.write_image(path, px.values.cast_dtype(sheet, dtype="uint8"), compression_level=6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="PNG output path")
    arguments = parser.parse_args()
    generate_sheet(arguments.output)


if __name__ == "__main__":
    main()
