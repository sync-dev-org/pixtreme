"""Uncompressed OpenEXR read lane."""

from __future__ import annotations

from collections.abc import Sequence

import cupy as cp
import numpy as np

from pixtreme._io.formats.exr.container import (
    _ExrChannel,
    _ExrContainer,
)
from pixtreme._io.formats.exr.packing import (
    _unpack_exr_output,
)


def _read_exr_none(
    container: _ExrContainer,
    selected: Sequence[_ExrChannel],
    *,
    output_dtype: str,
) -> cp.ndarray:
    host_file = np.frombuffer(container.data, dtype=np.uint8)
    decoded = cp.asarray(host_file)
    decoded_offsets = np.fromiter(
        (chunk.payload_start for chunk in container.chunks),
        dtype=np.int64,
        count=len(container.chunks),
    )
    decoded_sizes = np.fromiter(
        (chunk.expected_size for chunk in container.chunks),
        dtype=np.int64,
        count=len(container.chunks),
    )
    return _unpack_exr_output(
        container,
        selected,
        decoded,
        decoded_offsets,
        decoded_sizes,
        even_odd_grouped=np.zeros(len(container.chunks), dtype=np.uint8),
        output_dtype=output_dtype,
    )
