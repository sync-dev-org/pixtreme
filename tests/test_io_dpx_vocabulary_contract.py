"""Documentation contracts for the file-only DPX boundary."""

from __future__ import annotations

import pytest

from pixtreme._core.frame import _GAMMA_TOKENS
from pixtreme._io.formats import dpx


def test_vocabulary_documents_dpx_tokens_layout_storage_and_transfer_contracts(
    vocabulary_markdown: str,
) -> None:
    """v1-dpx acceptance 11: vocabulary and implementation agree on the complete DPX boundary."""
    dpx_boundary = " ".join(
        vocabulary_markdown.split("DPX is also file-only", maxsplit=1)[1]
        .split("\n## TIFF compression", maxsplit=1)[0]
        .split()
    )

    for fragment in (
        "`SDPX` or `XPDS`",
        "8-, 10-, 12-, or 16-bit",
        "Eight- and sixteen-bit samples require packing 0",
        "high end of a 32-bit word",
        "high 12 bits of a 16-bit word",
        "Method A filled",
        "One GPU pass resolves endian order",
        "`unchanged=True` returns 8-bit",
        "unique RGB or RGBA channels",
        "big-endian `SDPX`",
        "`bit_depth` defaults to 10",
        "records `cineon` as printing density",
    ):
        assert fragment in dpx_boundary

    accepted_depths = (8, 10, 12, 16)
    assert tuple(dpx._validate_dpx_bit_depth(depth) for depth in accepted_depths) == accepted_depths
    for invalid in (0, 9, 32, True, 10.0, "10"):
        with pytest.raises(ValueError):
            dpx._validate_dpx_bit_depth(invalid)
    assert "cineon" in _GAMMA_TOKENS
    assert {gamma: dpx._dpx_transfer_from_gamma(gamma) for gamma in ("cineon", "linear", "s-log3", "rec709")} == {
        "cineon": 1,
        "linear": 2,
        "s-log3": 3,
        "rec709": 6,
    }
