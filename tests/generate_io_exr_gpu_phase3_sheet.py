"""Generate the manual visual-acceptance sheets for v1 EXR GPU Phase 3."""

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
import pixtreme._io.formats.exr.selection as io
import pixtreme._io.header as io_header

_COMPRESSIONS = ("rle", "pxr24", "b44", "b44a")
_CHANNELS = ("R", "G", "B", "A", "diffuse.H")
_PANEL_LABEL_HEIGHT = 78
_DIFFERENCE_GAINS = {"rle": 65536.0, "pxr24": 4096.0, "b44": 16.0, "b44a": 16.0}


class _ComparisonMetric(TypedDict):
    max_abs: float
    rmse: float
    psnr_db: float | str


class _Metric(TypedDict):
    compression: str
    dtype: str
    reference_exr: str
    gpu_exr: str
    output_png: str
    compressed_chunks: int
    dense_blocks: int
    flat_blocks: int
    plinear_sections: int
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
    if height % 4 == 0:
        height -= 1
    if width % 4 == 0:
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
    values[96:160, 96:224, :] = np.asarray((0.25, 0.5, 1.5, 0.625, 0.375), dtype=np.float32)
    return values


def _codec_values(source: NDArray[np.float32], compression: str) -> NDArray[np.float32] | NDArray[np.float16]:
    if compression in {"b44", "b44a"}:
        return np.ascontiguousarray(source.astype(np.float16))
    return source


def _write_reference(
    path: Path,
    values: NDArray[np.float32] | NDArray[np.float16],
    compression: str,
) -> None:
    from openexr_dev_oracle import OpenEXR

    compression_value = {
        "rle": OpenEXR.RLE_COMPRESSION,
        "pxr24": OpenEXR.PXR24_COMPRESSION,
        "b44": OpenEXR.B44_COMPRESSION,
        "b44a": OpenEXR.B44A_COMPRESSION,
    }[compression]
    channels = {
        label: OpenEXR.Channel(
            label,
            np.ascontiguousarray(values[..., index]),
            1,
            1,
            compression in {"b44", "b44a"} and label == "diffuse.H",
        )
        for index, label in enumerate(_CHANNELS)
    }
    OpenEXR.File(
        {"type": OpenEXR.scanlineimage, "compression": compression_value},
        channels,
    ).write(str(path))


def _read_reference(path: Path) -> NDArray[np.float32] | NDArray[np.float16]:
    from openexr_dev_oracle import OpenEXR

    channels = OpenEXR.File(str(path), separate_channels=True).channels()
    return np.ascontiguousarray(np.stack([np.asarray(channels[label].pixels) for label in _CHANNELS], axis=2))


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


def _difference_rgb(left: NDArray[np.generic], right: NDArray[np.generic], *, gain: float) -> Image.Image:
    difference = np.max(
        np.abs(np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)),
        axis=2,
    )
    gained = np.clip(difference * np.float32(gain), np.float32(0.0), np.float32(1.0))
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
    compression: str,
    source: NDArray[np.generic],
    reference: NDArray[np.generic],
    gpu_read: NDArray[np.generic],
    custom_cpu_read: NDArray[np.generic],
    gpu_write: NDArray[np.generic],
    gpu_read_metric: _ComparisonMetric,
    custom_cpu_metric: _ComparisonMetric,
    gpu_write_metric: _ComparisonMetric,
) -> None:
    height, width, _ = source.shape
    gain = _DIFFERENCE_GAINS[compression]
    sheet = Image.new("RGB", (width * 8, height + _PANEL_LABEL_HEIGHT), color=(20, 20, 20))
    panels = (
        ("scene-linear source", _display_rgb(source)),
        ("OpenEXR reference", _display_rgb(reference)),
        ("GPU read", _display_rgb(gpu_read)),
        ("custom CPU read", _display_rgb(custom_cpu_read)),
        (f"GPU read abs diff x{gain:g}", _difference_rgb(reference, gpu_read, gain=gain)),
        (f"CPU read abs diff x{gain:g}", _difference_rgb(reference, custom_cpu_read, gain=gain)),
        ("GPU write pLinear=0 / OpenEXR read", _display_rgb(gpu_write)),
        (f"GPU write abs diff x{gain:g}", _difference_rgb(reference, gpu_write, gain=gain)),
    )
    draw = ImageDraw.Draw(sheet)
    for column, (label, panel) in enumerate(panels):
        left = column * width
        sheet.paste(panel, (left, _PANEL_LABEL_HEIGHT))
        draw.text((left + 8, 6), label, fill=(245, 245, 245))
    draw.text(
        (8, 28),
        f"{compression.upper()} {_CHANNELS!r} | {_metric_label('gpu-read', gpu_read_metric)}",
        fill=(205, 205, 205),
    )
    draw.text(
        (8, 49),
        f"{_metric_label('custom-read', custom_cpu_metric)} | {_metric_label('gpu-write', gpu_write_metric)}",
        fill=(205, 205, 205),
    )
    sheet.save(path)


def _restore_selection(key: tuple[str, str], original: object, sentinel: object) -> None:
    if original is sentinel:
        io._EXR_ROUTING.pop(key, None)
    else:
        io._EXR_ROUTING[key] = str(original)


def generate(source_path: Path, output_dir: Path) -> tuple[tuple[Path, ...], Path]:
    """Generate four codec sheets and a machine-readable metric manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source = _source_data(source_path)
    metrics: list[_Metric] = []
    output_paths: list[Path] = []
    for compression in _COMPRESSIONS:
        values = _codec_values(source, compression)
        reference_path = output_dir / f"exr-gpu-phase3-{compression}-openexr-reference.exr"
        gpu_path = output_dir / f"exr-gpu-phase3-{compression}-gpu-write.exr"
        sheet_path = output_dir / f"exr-gpu-phase3-{compression}.png"
        _write_reference(reference_path, values, compression)
        reference = _read_reference(reference_path)
        container = io_header._parse_exr(reference_path)
        compressed_descriptors = tuple(
            chunk.phase3 for chunk in container.chunks if chunk.phase3 is not None and not chunk.phase3.raw_stored
        )
        if not compressed_descriptors:
            raise AssertionError(f"{compression} visual fixture contains no compressed chunk")
        dense_blocks = sum(
            block.stored_size == 14 for descriptor in compressed_descriptors for block in descriptor.blocks
        )
        flat_blocks = sum(
            block.stored_size == 3 for descriptor in compressed_descriptors for block in descriptor.blocks
        )
        plinear_sections = sum(
            section.perceptually_linear
            for descriptor in compressed_descriptors
            for section in descriptor.channel_sections
        )
        if compression == "b44" and (dense_blocks == 0 or plinear_sections == 0):
            raise AssertionError("B44 visual fixture must contain dense and pLinear blocks")
        if compression == "b44a" and (dense_blocks == 0 or flat_blocks == 0 or plinear_sections == 0):
            raise AssertionError("B44A visual fixture must contain dense, flat, and pLinear blocks")

        frame = px.io.from_array(cp.asarray(values), colorspace="ACES2065-1", gamma="linear", channels=_CHANNELS)
        read_key = (compression, "read")
        write_key = (compression, "write")
        sentinel = object()
        original_read: object = io._EXR_ROUTING.get(read_key, sentinel)
        original_write: object = io._EXR_ROUTING.get(write_key, sentinel)
        try:
            io._EXR_ROUTING[read_key] = "gpu"
            gpu_read = cp.asnumpy(
                px.io.read_image(
                    reference_path,
                    channels=_CHANNELS,
                    unchanged=True,
                    colorspace="ACES2065-1",
                    gamma="linear",
                ).data
            )
            io._EXR_ROUTING[read_key] = "custom_cpu"
            custom_cpu_read = cp.asnumpy(
                px.io.read_image(
                    reference_path,
                    channels=_CHANNELS,
                    unchanged=True,
                    colorspace="ACES2065-1",
                    gamma="linear",
                ).data
            )
            io._EXR_ROUTING[write_key] = "gpu"
            px.io.write_image(gpu_path, frame, compression=compression)
        finally:
            _restore_selection(read_key, original_read, sentinel)
            _restore_selection(write_key, original_write, sentinel)
        gpu_write = _read_reference(gpu_path)
        data_range = float(np.max(values) - np.min(values))
        source_to_reference = _comparison_metric(values, reference, data_range=data_range)
        gpu_read_metric = _comparison_metric(reference, gpu_read, data_range=data_range)
        custom_cpu_metric = _comparison_metric(reference, custom_cpu_read, data_range=data_range)
        gpu_write_metric = _comparison_metric(reference, gpu_write, data_range=data_range)
        _write_sheet(
            sheet_path,
            compression=compression,
            source=values,
            reference=reference,
            gpu_read=gpu_read,
            custom_cpu_read=custom_cpu_read,
            gpu_write=gpu_write,
            gpu_read_metric=gpu_read_metric,
            custom_cpu_metric=custom_cpu_metric,
            gpu_write_metric=gpu_write_metric,
        )
        metrics.append(
            {
                "compression": compression,
                "dtype": str(values.dtype),
                "reference_exr": reference_path.name,
                "gpu_exr": gpu_path.name,
                "output_png": sheet_path.name,
                "compressed_chunks": len(compressed_descriptors),
                "dense_blocks": dense_blocks,
                "flat_blocks": flat_blocks,
                "plinear_sections": plinear_sections,
                "source_to_openexr_reference": source_to_reference,
                "openexr_reference_to_gpu_read": gpu_read_metric,
                "openexr_reference_to_custom_cpu_read": custom_cpu_metric,
                "openexr_reference_to_gpu_write": gpu_write_metric,
            }
        )
        output_paths.extend((reference_path, gpu_path, sheet_path))

    manifest_path = output_dir / "exr-gpu-phase3-metrics.json"
    manifest = {
        "source": {
            "path": str(source_path),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "description": "MtTamWest.exr from the official OpenEXR sample-image repository",
        },
        "fixture": {
            "description": (
                "scene-linear photograph with negative and >1 values, alpha, dotted pLinear reference-read channel, "
                "public-write pLinear=0, constant and dense blocks, and partial 4x4/32-row edges"
            ),
            "shape": list(source.shape),
            "channels": list(_CHANNELS),
            "difference_gains": _DIFFERENCE_GAINS,
        },
        "metrics": metrics,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return tuple(output_paths), manifest_path


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
