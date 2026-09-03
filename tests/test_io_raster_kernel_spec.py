"""Characterization tests for fused raster boundary repacking."""

from __future__ import annotations

import inspect

import cupy as cp
import numpy as np
import pytest

from pixtreme._core.frame import Frame
from pixtreme._io.dtype import _prepare_write_frame
from pixtreme._io.formats.nvimgcodec import _decode_raster_data, _raster_write_data, _repack_raster_data


def _assert_bit_equal(actual: cp.ndarray, expected: cp.ndarray) -> None:
    assert actual.dtype == expected.dtype
    unsigned_dtype = f"uint{actual.dtype.itemsize * 8}"
    cp.testing.assert_array_equal(actual.view(unsigned_dtype), expected.view(unsigned_dtype))


def _raster_values(dtype: type[np.generic], component_count: int) -> cp.ndarray:
    if np.issubdtype(dtype, np.integer):
        maximum = int(np.iinfo(dtype).max)
        values = np.asarray((0, 1, maximum // 4, maximum // 2, maximum - 1, maximum), dtype=dtype)
    else:
        values = np.asarray((-np.inf, -1.0, -0.0, 0.0, 0.25, 0.5, 1.0, np.inf, np.nan), dtype=dtype)
    return cp.asarray(np.resize(values, 5 * 7 * component_count).reshape(5, 7, component_count))


def _composed_decode(source: cp.ndarray, indices: tuple[int, ...], *, unchanged: bool) -> cp.ndarray:
    output = source[..., list(indices)]
    if not unchanged:
        if output.dtype.name in {"uint8", "uint16"}:
            output = output.astype(cp.float32) / np.float32(np.iinfo(output.dtype).max)
        elif output.dtype.name == "float16":
            output = output.astype(cp.float32)
    return cp.ascontiguousarray(output)


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.float16, np.float32), ids=np.dtype)
@pytest.mark.parametrize(
    ("component_count", "indices"),
    ((1, (0,)), (3, (2, 0)), (4, (3, 1, 0))),
)
@pytest.mark.parametrize("unchanged", (True, False), ids=("unchanged", "normalized"))
def test_fused_decode_repack_is_bit_identical_to_the_composed_boundary_characterization(
    dtype: type[np.generic],
    component_count: int,
    indices: tuple[int, ...],
    unchanged: bool,
) -> None:
    """characterization: GitHub issue #1 RawKernel trial, raster-decode acceptance 1 and 5 freezes decode bits."""
    source = _raster_values(dtype, component_count)
    expected = _composed_decode(source, indices, unchanged=unchanged)

    actual = _decode_raster_data(source, indices, unchanged=unchanged)

    assert actual.flags.c_contiguous
    _assert_bit_equal(actual, expected)


def test_fused_decode_repack_is_bit_identical_for_strided_source_storage_characterization() -> None:
    """characterization: GitHub issue #1 RawKernel trial, raster-decode acceptance 1 and 5 freezes strided bits."""
    source = _raster_values(np.uint16, 4)[:, ::2, :]
    indices = (3, 1, 0)
    expected = _composed_decode(source, indices, unchanged=False)

    actual = _decode_raster_data(source, indices, unchanged=False)

    _assert_bit_equal(actual, expected)


def _composed_write(frame: Frame, format_name: str) -> cp.ndarray:
    prepared = _prepare_write_frame(format_name, frame)
    canonical = {
        frozenset(("R", "G", "B")): ("R", "G", "B"),
        frozenset(("R", "G", "B", "A")): ("R", "G", "B", "A"),
        frozenset(("Y",)): ("Y",),
        frozenset(("Y", "A")): ("Y", "A"),
    }[frozenset(frame.channels)]
    indices = [frame.channels.index(label) for label in canonical]
    return cp.ascontiguousarray(prepared.data[..., indices])


@pytest.mark.parametrize(
    ("format_name", "dtype", "channels"),
    (
        ("PNG", np.uint8, ("B", "R", "G")),
        ("TIFF", np.uint16, ("A", "B", "G", "R")),
        ("JPEG", np.uint16, ("G", "B", "R")),
        ("WEBP", np.uint32, ("B", "R", "G")),
        ("PNG", np.float16, ("G", "R", "B")),
        ("TIFF", np.float32, ("B", "G", "R")),
    ),
    ids=("native-u8", "native-u16", "u16-u8", "u32-u8", "f16-u8", "f32-u8"),
)
def test_fused_write_repack_is_bit_identical_to_recode_then_canonical_gather_characterization(
    format_name: str,
    dtype: type[np.generic],
    channels: tuple[str, ...],
) -> None:
    """characterization: GitHub issue #1 RawKernel trial, raster-decode acceptance 2 and 5 freezes write bits."""
    frame = Frame(
        data=_raster_values(dtype, len(channels)),
        colorspace="ACEScg",
        gamma="linear",
        channels=channels,
    )
    expected = _composed_write(frame, format_name)

    actual = _raster_write_data(frame, format_name=format_name)

    assert actual.flags.c_contiguous
    _assert_bit_equal(actual, expected)


def test_raster_boundary_helpers_route_through_the_single_pass_repack_launcher_characterization() -> None:
    """characterization: GitHub issue #1 RawKernel trial, raster-decode acceptance 1, 2, and 5 freezes one pass."""
    for function in (_decode_raster_data, _raster_write_data):
        source = inspect.getsource(function)
        assert "_repack_raster_data(" in source
        assert "astype(" not in source
        assert "ascontiguousarray(" not in source
    assert inspect.getsource(_repack_raster_data).count("_raster_repack_kernel(") == 1
