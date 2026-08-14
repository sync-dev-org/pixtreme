"""Documentation contracts for the file-only DPX boundary."""

from __future__ import annotations

import pytest

from pixtreme._core.frame import _GAMMA_TOKENS
from pixtreme._io.formats import dpx


def test_vocabulary_documents_dpx_tokens_layout_storage_and_transfer_contracts(
    vocabulary_markdown: str,
) -> None:
    """v1-dpx acceptance 11: vocabulary and implementation agree on the complete DPX boundary."""
    dpx_boundary = vocabulary_markdown.split("DPX も file-only format", maxsplit=1)[1].split(
        "\n## TIFF compression", maxsplit=1
    )[0]

    for fragment in (
        "`SDPX` / `XPDS`",
        "8 / 10 / 12 / 16-bit",
        "8 / 16-bit は packing 0",
        "10-bit は 32-bit word",
        "12-bit は 16-bit word",
        "Method A filled",
        "GPU 1 pass で endian 解決",
        "`unchanged=True` は 8-bit を uint8、10 / 12 / 16-bit code を uint16",
        "unique RGB / RGBA Frame",
        "big-endian `SDPX`",
        "`bit_depth` の既定は 10",
        "Frame gamma は `cineon`→printing density",
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
