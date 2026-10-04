"""EXR payload selection shared by compressed write paths."""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import pixtreme._io.formats.exr.packing as exr_packing


@pytest.mark.req("REQ-PIX-007")
@pytest.mark.parametrize("compression", ("rle", "pxr24", "b44", "b44a"))
def test_exr_write_stores_raw_chunks_when_compression_does_not_shrink_them(compression: str) -> None:
    """EXR writes store raw chunks when compression yields an equal or larger payload."""
    assert compression in {"rle", "pxr24", "b44", "b44a"}
    raw_chunks = (b"raw!", b"same", b"last")
    encoded_chunks = (b"zip", b"code", b"larger")
    raw_sizes = tuple(map(len, raw_chunks))
    encoded_sizes = tuple(map(len, encoded_chunks))
    raw_offsets = tuple(int(value) for value in np.cumsum((0, *raw_sizes[:-1]), dtype=np.int64))
    encoded_offsets = tuple(int(value) for value in np.cumsum((0, *encoded_sizes[:-1]), dtype=np.int64))

    selected, selected_sizes = exr_packing._select_exr_payloads(
        cp.asarray(np.frombuffer(b"".join(raw_chunks), dtype=np.uint8)),
        raw_offsets,
        raw_sizes,
        cp.asarray(np.frombuffer(b"".join(encoded_chunks), dtype=np.uint8)),
        encoded_offsets,
        encoded_sizes,
    )

    assert selected_sizes == (3, 4, 4)
    assert selected.get().tobytes() == b"zip" + raw_chunks[1] + raw_chunks[2]
