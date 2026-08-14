"""Generate the manual visual-acceptance sheet for v1 EXR GPU Phase 4 PIZ."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import TypedDict

import cupy as cp
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

import pixtreme as px
import pixtreme._io.formats.exr.container as exr_container
import pixtreme._io.formats.exr.selection as io
import pixtreme._io.header as io_header

_SELECTED_CHANNELS = ("R", "G", "B", "A", "diffuse.H")
_UINT_CHANNEL = "mask.U"
_PANEL_LABEL_HEIGHT = 78
_DIFFERENCE_GAIN = 65536.0


class _ComparisonMetric(TypedDict):
    max_abs: float
    rmse: float
    psnr_db: float | str


class _Metric(TypedDict):
    source_to_openexr_reference: _ComparisonMetric
    openexr_reference_to_gpu_read: _ComparisonMetric
    openexr_reference_to_custom_cpu_read: _ComparisonMetric
    openexr_reference_to_gpu_write: _ComparisonMetric


def _source_data(source_path: Path) -> NDArray[np.float32]:
    from openexr_dev_oracle import OpenEXR

    source = OpenEXR.File(str(source_path), separate_channels=True).channels()
    rgb = np.stack([np.asarray(source[label].pixels, dtype=np.float32) for label in "RGB"], axis=2)
    step = max(1, int(np.ceil(max(rgb.shape[0] / 365, rgb.shape[1] / 607))))
    rgb = np.ascontiguousarray(rgb[::step, ::step])
    height = min(365, rgb.shape[0])
    width = min(607, rgb.shape[1])
    if height % 32 == 0:
        height -= 1
    if width % 2 == 0:
        width -= 1
    rgb = np.nan_to_num(rgb[:height, :width], nan=0.0, posinf=65504.0, neginf=-65504.0)
    rgb = np.ascontiguousarray(rgb * np.float32(1.125) - np.float32(0.015))
    rgb[0, 0] = np.asarray((-0.125, 1.25, 2.0), dtype=np.float32)
    x = np.linspace(np.float32(0.0), np.float32(1.0), width, dtype=np.float32)
    y = np.linspace(np.float32(0.0), np.float32(1.0), height, dtype=np.float32)
    alpha = np.ascontiguousarray(np.minimum(x[None, :], y[:, None]) * np.float32(0.85) + np.float32(0.1))
    dotted = np.ascontiguousarray(rgb[..., 0] * np.float32(0.25) - np.float32(0.03))
    values = np.ascontiguousarray(np.concatenate((rgb, alpha[..., None], dotted[..., None]), axis=2))
    values[32:48, :, :] = np.asarray((0.125, 0.5, 1.25, 0.75, 0.25), dtype=np.float32)
    return values


def _uint_data(shape: tuple[int, int]) -> NDArray[np.uint32]:
    height, width = shape
    y, x = np.mgrid[:height, :width]
    return np.ascontiguousarray(
        x.astype(np.uint32) * np.uint32(0x9E3779B1) + y.astype(np.uint32) * np.uint32(0x85EBCA77)
    )


def _write_reference(
    path: Path,
    values: NDArray[np.float32],
    unsigned: NDArray[np.uint32],
) -> None:
    from openexr_dev_oracle import OpenEXR

    channels: dict[str, NDArray[np.generic]] = {
        label: np.ascontiguousarray(values[..., index]) for index, label in enumerate(_SELECTED_CHANNELS)
    }
    channels[_UINT_CHANNEL] = unsigned
    OpenEXR.File(
        {"type": OpenEXR.scanlineimage, "compression": OpenEXR.PIZ_COMPRESSION},
        channels,
    ).write(str(path))


def _read_reference(path: Path) -> NDArray[np.float32]:
    from openexr_dev_oracle import OpenEXR

    channels = OpenEXR.File(str(path), separate_channels=True).channels()
    return np.ascontiguousarray(
        np.stack([np.asarray(channels[label].pixels, dtype=np.float32) for label in _SELECTED_CHANNELS], axis=2)
    )


def _assert_visual_fixture(path: Path, *, require_uint: bool) -> tuple[int, int, int]:
    container = io_header._parse_exr(path)
    if not container.piz_eligible:
        raise AssertionError(f"{path} is not an eligible PIZ visual fixture")
    descriptors = tuple(chunk.piz for chunk in container.chunks if chunk.piz is not None)
    compressed = tuple(descriptor for descriptor in descriptors if not descriptor.raw_stored)
    partial = tuple(descriptor for descriptor in descriptors if descriptor.row_count < descriptor.lines_per_chunk)
    if not compressed or not partial:
        raise AssertionError(f"{path} must contain compressed PIZ data and a partial final chunk")
    if require_uint and not any(
        plane.channel_name == _UINT_CHANNEL for descriptor in descriptors for plane in descriptor.channel_planes
    ):
        raise AssertionError(f"{path} contains no unselected UINT channel")
    nonempty_bitmaps = 0
    huffman_data_bytes = 0
    for descriptor in compressed:
        bitmap = container.data[descriptor.bitmap_span.start : descriptor.bitmap_span.end]
        nonempty_bitmaps += int(any(bitmap))
        stream = container.data[descriptor.huffman_span.start : descriptor.huffman_span.end]
        table = exr_container._parse_piz_huffman_table(stream)
        huffman_data_bytes += table.data_span.size
    if nonempty_bitmaps == 0 or huffman_data_bytes == 0:
        raise AssertionError(f"{path} contains no nonempty bitmap or Huffman data")
    return len(compressed), len(partial), nonempty_bitmaps


def _display_rgb(values: NDArray[np.generic]) -> Image.Image:
    positive = np.maximum(np.asarray(values[..., :3], dtype=np.float32), np.float32(0.0))
    mapped = positive / (np.float32(1.0) + positive)
    srgb = np.where(
        mapped <= np.float32(0.0031308),
        mapped * np.float32(12.92),
        np.float32(1.055) * np.power(mapped, np.float32(1.0 / 2.4)) - np.float32(0.055),
    )
    code = np.asarray(np.floor(np.clip(srgb, 0.0, 1.0) * np.float32(255.0) + np.float32(0.5)), dtype=np.uint8)
    return Image.fromarray(code, mode="RGB")


def _difference_rgb(left: NDArray[np.generic], right: NDArray[np.generic]) -> Image.Image:
    difference = np.max(
        np.abs(np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)),
        axis=2,
    )
    gained = np.clip(difference * np.float32(_DIFFERENCE_GAIN), np.float32(0.0), np.float32(1.0))
    code = np.asarray(np.floor(gained * np.float32(255.0) + np.float32(0.5)), dtype=np.uint8)
    return Image.fromarray(np.stack((code, code // 4, np.zeros_like(code)), axis=2), mode="RGB")


def _comparison_metric(
    reference: NDArray[np.generic],
    candidate: NDArray[np.generic],
    *,
    data_range: float,
) -> _ComparisonMetric:
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    max_abs = float(np.max(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    psnr: float | str = "infinity" if rmse == 0.0 else float(20.0 * math.log10(data_range / rmse))
    return {"max_abs": max_abs, "rmse": rmse, "psnr_db": psnr}


def _metric_label(name: str, metric: _ComparisonMetric) -> str:
    psnr = metric["psnr_db"]
    psnr_text = psnr if isinstance(psnr, str) else f"{psnr:.3f}dB"
    return f"{name}: max={metric['max_abs']:.6g} rmse={metric['rmse']:.6g} psnr={psnr_text}"


def _write_sheet(
    path: Path,
    *,
    source: NDArray[np.float32],
    reference: NDArray[np.float32],
    gpu_read: NDArray[np.float32],
    custom_cpu_read: NDArray[np.float32],
    gpu_write: NDArray[np.float32],
    metrics: _Metric,
) -> None:
    height, width, _ = source.shape
    sheet = Image.new("RGB", (width * 8, height + _PANEL_LABEL_HEIGHT), color=(20, 20, 20))
    panels = (
        ("scene-linear source", _display_rgb(source)),
        ("OpenEXR PIZ reference", _display_rgb(reference)),
        ("GPU read", _display_rgb(gpu_read)),
        ("custom CPU read", _display_rgb(custom_cpu_read)),
        (f"GPU read abs diff x{_DIFFERENCE_GAIN:g}", _difference_rgb(reference, gpu_read)),
        (f"CPU read abs diff x{_DIFFERENCE_GAIN:g}", _difference_rgb(reference, custom_cpu_read)),
        ("GPU PIZ write / OpenEXR read", _display_rgb(gpu_write)),
        (f"GPU write abs diff x{_DIFFERENCE_GAIN:g}", _difference_rgb(reference, gpu_write)),
    )
    draw = ImageDraw.Draw(sheet)
    for column, (label, panel) in enumerate(panels):
        left = column * width
        sheet.paste(panel, (left, _PANEL_LABEL_HEIGHT))
        draw.text((left + 8, 6), label, fill=(245, 245, 245))
    draw.text(
        (8, 28),
        f"PIZ {_SELECTED_CHANNELS!r} + unselected {_UINT_CHANNEL} | "
        f"{_metric_label('gpu-read', metrics['openexr_reference_to_gpu_read'])}",
        fill=(205, 205, 205),
    )
    draw.text(
        (8, 49),
        f"{_metric_label('custom-read', metrics['openexr_reference_to_custom_cpu_read'])} | "
        f"{_metric_label('gpu-write', metrics['openexr_reference_to_gpu_write'])}",
        fill=(205, 205, 205),
    )
    sheet.save(path)


def _restore_selection(key: tuple[str, str], original: object, sentinel: object) -> None:
    if original is sentinel:
        io._EXR_ROUTING.pop(key, None)
    else:
        io._EXR_ROUTING[key] = str(original)


def generate(source_path: Path, output_dir: Path) -> tuple[tuple[Path, ...], Path]:
    """Generate the PIZ comparison sheet, EXR intermediates, and metric manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source = _source_data(source_path)
    unsigned = _uint_data(source.shape[:2])
    reference_path = output_dir / "exr-gpu-phase4-piz-openexr-reference.exr"
    gpu_path = output_dir / "exr-gpu-phase4-piz-gpu-write.exr"
    sheet_path = output_dir / "exr-gpu-phase4-piz.png"
    _write_reference(reference_path, source, unsigned)
    reference = _read_reference(reference_path)
    reference_compressed, reference_partial, reference_bitmaps = _assert_visual_fixture(
        reference_path,
        require_uint=True,
    )

    frame = px.io.from_array(cp.asarray(source), colorspace="ACES2065-1", gamma="linear", channels=_SELECTED_CHANNELS)
    read_key = ("piz", "read")
    write_key = ("piz", "write")
    sentinel = object()
    original_read: object = io._EXR_ROUTING.get(read_key, sentinel)
    original_write: object = io._EXR_ROUTING.get(write_key, sentinel)
    try:
        io._EXR_ROUTING[read_key] = "gpu"
        gpu_read = cp.asnumpy(
            px.io.read_image(
                reference_path,
                channels=_SELECTED_CHANNELS,
                unchanged=True,
                colorspace="ACES2065-1",
                gamma="linear",
            ).data
        )
        io._EXR_ROUTING[read_key] = "custom_cpu"
        custom_cpu_read = cp.asnumpy(
            px.io.read_image(
                reference_path,
                channels=_SELECTED_CHANNELS,
                unchanged=True,
                colorspace="ACES2065-1",
                gamma="linear",
            ).data
        )
        io._EXR_ROUTING[write_key] = "gpu"
        px.io.write_image(gpu_path, frame, compression="piz")
    finally:
        _restore_selection(read_key, original_read, sentinel)
        _restore_selection(write_key, original_write, sentinel)
    gpu_write = _read_reference(gpu_path)
    gpu_compressed, gpu_partial, gpu_bitmaps = _assert_visual_fixture(gpu_path, require_uint=False)

    data_range = float(np.max(source) - np.min(source))
    metrics: _Metric = {
        "source_to_openexr_reference": _comparison_metric(source, reference, data_range=data_range),
        "openexr_reference_to_gpu_read": _comparison_metric(reference, gpu_read, data_range=data_range),
        "openexr_reference_to_custom_cpu_read": _comparison_metric(
            reference,
            custom_cpu_read,
            data_range=data_range,
        ),
        "openexr_reference_to_gpu_write": _comparison_metric(reference, gpu_write, data_range=data_range),
    }
    _write_sheet(
        sheet_path,
        source=source,
        reference=reference,
        gpu_read=gpu_read,
        custom_cpu_read=custom_cpu_read,
        gpu_write=gpu_write,
        metrics=metrics,
    )

    manifest_path = output_dir / "exr-gpu-phase4-piz-metrics.json"
    manifest = {
        "source": {
            "path": str(source_path),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "description": "MtTamWest.exr from the official OpenEXR sample-image repository",
        },
        "fixture": {
            "description": (
                "scene-linear photograph with negative and >1 values, alpha, dotted channel, unselected UINT, "
                "and a partial 32-row PIZ chunk"
            ),
            "shape": list(source.shape),
            "selected_channels": list(_SELECTED_CHANNELS),
            "unselected_uint_channel": _UINT_CHANNEL,
            "difference_gain": _DIFFERENCE_GAIN,
            "reference_compressed_chunks": reference_compressed,
            "reference_partial_chunks": reference_partial,
            "reference_nonempty_bitmaps": reference_bitmaps,
            "gpu_write_compressed_chunks": gpu_compressed,
            "gpu_write_partial_chunks": gpu_partial,
            "gpu_write_nonempty_bitmaps": gpu_bitmaps,
        },
        "artifacts": {
            "reference_exr": reference_path.name,
            "gpu_exr": gpu_path.name,
            "output_png": sheet_path.name,
        },
        "metrics": metrics,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return (reference_path, gpu_path, sheet_path), manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-exr", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    output_paths, manifest_path = generate(arguments.source_exr, arguments.output_dir)
    for path in (*output_paths, manifest_path):
        print(path)


if __name__ == "__main__":
    main()
