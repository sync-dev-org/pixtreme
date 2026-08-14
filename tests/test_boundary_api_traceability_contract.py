"""Traceability contracts for the generic device-array boundary."""

from __future__ import annotations

import inspect

import pixtreme as px


def _section(markdown: str, heading: str) -> str:
    return markdown.split(f"## {heading}\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]


def test_frame_has_no_numpy_exit_and_vocabulary_names_explicit_host_transfers(
    vocabulary_markdown: str,
) -> None:
    """v1-boundary-api acceptance 18: Frame has no NumPy method and docs name explicit host exits."""
    assert not hasattr(px.core.Frame, "to_numpy")

    boundary = _section(vocabulary_markdown, "Frame boundary contract")
    assert "`px.io.to_array(frame, ...).get()`" in boundary
    assert "`cp.asnumpy(px.io.to_array(frame, ...))`" in boundary


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
