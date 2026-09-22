"""Traceability contracts for the generic device-array boundary."""

from __future__ import annotations

import inspect

import pixtreme as px


def test_array_boundary_docstrings_state_copy_out_dlpack_and_inverse_affine_contracts() -> None:
    """v1-boundary-api acceptance 22: array-boundary docstrings expose every LLM-readable contract axis."""
    to_doc = " ".join((inspect.getdoc(px.io.to_array) or "").split())
    from_doc = " ".join((inspect.getdoc(px.io.from_array) or "").split())

    for fragment in (
        "``y = (x * scale - mean) / std``",
        "``copy=None`` uses a zero-copy view",
        "``copy=False`` strictly requires zero-copy",
        "``copy=True`` always returns private storage",
        "With ``out``, copy must be omitted",
        "exactly shaped, exactly typed, C-contiguous ``cupy.ndarray``",
        "non-CuPy DLPack producer is intentionally rejected",
        "The returned ``cupy.ndarray`` is itself a DLPack producer",
        "Frame is also a DLPack producer",
    ):
        assert fragment in to_doc
    for fragment in (
        "``x = (y * std + mean) / scale``",
        "round-trips :func:`pixtreme.io.to_array`",
        "``copy=None`` retains a zero-copy",
        "``copy=False`` strictly guarantees zero-copy",
        "``copy=True`` always gives the Frame private storage",
        "Host arrays and CPU DLPack producers are rejected",
    ):
        assert fragment in from_doc
