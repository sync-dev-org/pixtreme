"""Specification tests for native UINT lanes in the non-PIZ OpenEXR codecs."""

from __future__ import annotations

import struct
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import pixtreme._io.formats.exr.codec_b44 as codec_b44
import pixtreme._io.formats.exr.codec_dwa as codec_dwa
import pixtreme._io.formats.exr.codec_none as codec_none
import pixtreme._io.formats.exr.codec_pxr24 as codec_pxr24
import pixtreme._io.formats.exr.codec_rle as codec_rle
import pixtreme._io.formats.exr.codec_zip as codec_zip
import pixtreme._io.formats.exr.container as container_module
import pixtreme._io.formats.exr.selection as selection

_CHROMATICITIES = (0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.077, 0.32168, 0.33767)


def _pack_uint_gpu(
    data: object,
    channels: Sequence[str],
    *,
    row_prefix_bytes: int = 0,
) -> tuple[object, tuple[str, ...], int]:
    """Independent scope-boundary packer preserving native little-endian UINT samples."""
    import cupy as cp

    values = cp.asarray(data)
    assert values.dtype == cp.uint32
    assert values.ndim == 3
    assert row_prefix_bytes in (0, 8)
    ordered = tuple(sorted(enumerate(channels), key=lambda item: item[1]))
    ordered_channels = tuple(name for _, name in ordered)
    indices = cp.asarray(np.asarray([index for index, _ in ordered], dtype=np.int32))
    planar = cp.ascontiguousarray(values[:, :, indices].transpose(0, 2, 1)).view(cp.uint8).reshape(-1)
    if not row_prefix_bytes:
        return planar, ordered_channels, 0

    height, width, channel_count = (int(value) for value in values.shape)
    row_bytes = width * channel_count * 4
    prefixes = np.frombuffer(
        b"".join(struct.pack("<ii", row, row_bytes) for row in range(height)),
        dtype=np.uint8,
    ).reshape(height, row_prefix_bytes)
    records = cp.empty((height, row_prefix_bytes + row_bytes), dtype=cp.uint8)
    records[:, :row_prefix_bytes] = cp.asarray(prefixes)
    records[:, row_prefix_bytes:] = planar.reshape(height, row_bytes)
    return records.reshape(-1), ordered_channels, 0


def _uint_samples(*, height: int) -> np.ndarray:
    values = np.zeros((height, 8), dtype=np.uint32)
    boundary = np.asarray(
        (0x00000000, 0x00000001, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0x01020304, 0xA5A5A5A5, 0xDEADBEEF),
        dtype=np.uint32,
    )
    values[0] = boundary
    values[-1] = boundary[::-1]
    if height > 2:
        values[height // 2] = boundary ^ np.uint32(0xFFFFFFFF)
    return values


def _internal_reader(compression: str) -> Callable[..., object]:
    if compression == "none":
        return codec_none._read_exr_none
    if compression in ("zip", "zips"):
        return codec_zip._read_exr_zip_custom_cpu
    if compression == "rle":
        return codec_rle._read_exr_rle_gpu
    if compression == "pxr24":
        return codec_pxr24._read_exr_pxr24_custom_cpu
    if compression in ("b44", "b44a"):
        return codec_b44._read_exr_b44_gpu
    return codec_dwa._read_exr_dwa_gpu


def _with_codec_chunks(container: container_module._ExrContainer) -> container_module._ExrContainer:
    """Materialize UINT NONE/ZIP descriptors before the Phase 4 container gate opens them."""
    if container.chunks:
        return container
    part = container.parts[0]
    header_end = max(attribute.payload_end for attribute in part.attributes.values()) + 1
    offset_table, chunks = container_module._parse_candidate_chunks(
        container.data,
        header_end,
        part,
        lines_per_chunk=container.lines_per_chunk,
    )
    materialized_part = replace(part, offset_table=offset_table, chunks=chunks)
    return replace(container, parts=(materialized_part,), offset_table=offset_table, chunks=chunks)


@pytest.mark.parametrize(
    "compression",
    ("none", "zips", "zip", "rle", "pxr24", "b44", "b44a", "dwaa", "dwab"),
)
def test_non_piz_codec_uint_write_and_read_lanes_preserve_sample_bits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compression: str,
) -> None:
    """v1-exr-runtime-independence acceptance 30 and 31: codec-direct UINT writes and reads are bit exact."""
    import cupy as cp
    from openexr_dev_oracle import OpenEXR

    height = 257 if compression == "dwab" else 33
    samples = _uint_samples(height=height)
    data = cp.asarray(samples[..., np.newaxis])
    path = tmp_path / f"uint-{compression}.exr"
    channel_name = "A" if compression in ("dwaa", "dwab") else "Y"
    monkeypatch.setattr(selection, "_pack_exr_gpu", _pack_uint_gpu)
    monkeypatch.setattr(codec_dwa, "_pack_exr_gpu", _pack_uint_gpu)

    if compression in ("dwaa", "dwab"):
        codec_dwa._write_exr_dwa_gpu(
            path,
            data,
            (channel_name,),
            compression=compression,
            dwa_level=45.0,
            chromaticities=_CHROMATICITIES,
            aces_image_container=True,
        )
    else:
        selection._write_exr_gpu(
            path,
            data,
            (channel_name,),
            compression=compression,
            chromaticities=_CHROMATICITIES,
            aces_image_container=True,
        )

    reference_file = OpenEXR.File(str(path), separate_channels=True)
    reference = np.asarray(reference_file.channels()[channel_name].pixels)
    assert reference.dtype == np.uint32
    np.testing.assert_array_equal(reference, samples)

    container = _with_codec_chunks(container_module._parse_exr_container(path))
    selected = (container.parts[0].channels[0],)
    assert selected[0].pixel_type == 0
    if compression in ("dwaa", "dwab"):
        layouts = tuple(
            chunk.dwa.channel_layout
            for chunk in container.chunks
            if not chunk.raw_stored and chunk.dwa is not None and chunk.dwa.channel_layout is not None
        )
        assert layouts
        assert {descriptor.scheme for layout in layouts for descriptor in layout.channels} == {"unknown"}
    actual = _internal_reader(compression)(container, selected, output_dtype="uint32")

    assert actual.dtype == cp.uint32
    np.testing.assert_array_equal(actual.get()[..., 0], samples)


@pytest.mark.parametrize("compression", ("b44", "b44a"))
def test_b44_compressed_mixed_file_reads_raw_uint_section_as_native_uint32(
    tmp_path: Path,
    compression: str,
) -> None:
    """v1-exr-runtime-independence acceptance 30: B44 raw UINT sections survive compressed mixed-type chunks."""
    import cupy as cp
    from openexr_dev_oracle import OpenEXR

    height, width = 7, 8
    unsigned = _uint_samples(height=height)[:, :width]
    half = np.full((height, width), np.float16(0.5), dtype=np.float16)
    reference_compression = OpenEXR.B44_COMPRESSION if compression == "b44" else OpenEXR.B44A_COMPRESSION
    path = tmp_path / f"mixed-{compression}-uint.exr"
    OpenEXR.File({"compression": reference_compression}, {"H": half, "U": unsigned}).write(str(path))

    container = container_module._parse_exr_container(path)
    selected = tuple(channel for channel in container.parts[0].channels if channel.name == "U")
    assert selected[0].pixel_type == 0
    assert any(not chunk.raw_stored for chunk in container.chunks)

    actual = codec_b44._read_exr_b44_gpu(container, selected, output_dtype="uint32")

    assert actual.dtype == cp.uint32
    np.testing.assert_array_equal(actual.get()[..., 0], unsigned)
