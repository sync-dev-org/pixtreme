"""Documentation contracts for the file-only DPX boundary."""

from __future__ import annotations

import pytest

from pixtreme._core.frame import _GAMMA_TOKENS
from pixtreme._io.formats import dpx


def test_vocabulary_documents_dpx_tokens_layout_storage_and_transfer_contracts(
    vocabulary_markdown: str,
) -> None:
    """v1-dpx acceptance 11; v1-sony-tokens acceptance 12; v1-arri-tokens acceptance 27 and 29;
    v1-blackmagic-tokens acceptance 48 and 50; v1-red-tokens acceptance 70 and 72;
    v1-canon-tokens acceptance 91 and 93; v1-panasonic-tokens acceptance 110 and 112;
    v1-standard-tokens acceptance 133 and 135; v1-vendor-a-tokens acceptance 159 and 161;
    v1-vendor-b-tokens acceptance 186 and 188.

    Docs and code agree on the DPX boundary.
    """
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
        "records `Cineon` and `REDlogFilm` as printing density",
        "`D-Log`, `F-Log`, `F-Log2`, `N-Log`, `L-Log`, `Apple-Log`, `Samsung-Log`, `ACEScc`, and `ACEScct` as logarithmic",
        "including `Gamma-2.5` as the BT.709 transfer",
    ):
        assert fragment in dpx_boundary

    accepted_depths = (8, 10, 12, 16)
    assert tuple(dpx._validate_dpx_bit_depth(depth) for depth in accepted_depths) == accepted_depths
    for invalid in (0, 9, 32, True, 10.0, "10"):
        with pytest.raises(ValueError):
            dpx._validate_dpx_bit_depth(invalid)
    assert "Cineon" in _GAMMA_TOKENS
    assert {
        gamma: dpx._dpx_transfer_from_gamma(gamma)
        for gamma in (
            "Cineon",
            "linear",
            "S-Log",
            "S-Log2",
            "S-Log3",
            "ARRI-LogC3",
            "ARRI-LogC4",
            "Blackmagic-Film-Gen-5",
            "DaVinci-Intermediate",
            "RED-Log3G10",
            "REDlogFilm",
            "Canon-Log",
            "Canon-Log-2",
            "Canon-Log-3",
            "V-Log",
            "D-Log",
            "F-Log",
            "F-Log2",
            "N-Log",
            "L-Log",
            "Apple-Log",
            "Samsung-Log",
            "ACEScc",
            "ACEScct",
            "Rec.709",
            "Gamma-2.5",
        )
    } == {
        "Cineon": 1,
        "linear": 2,
        "S-Log": 3,
        "S-Log2": 3,
        "S-Log3": 3,
        "ARRI-LogC3": 3,
        "ARRI-LogC4": 3,
        "Blackmagic-Film-Gen-5": 3,
        "DaVinci-Intermediate": 3,
        "RED-Log3G10": 3,
        "REDlogFilm": 1,
        "Canon-Log": 3,
        "Canon-Log-2": 3,
        "Canon-Log-3": 3,
        "V-Log": 3,
        "D-Log": 3,
        "F-Log": 3,
        "F-Log2": 3,
        "N-Log": 3,
        "L-Log": 3,
        "Apple-Log": 3,
        "Samsung-Log": 3,
        "ACEScc": 3,
        "ACEScct": 3,
        "Rec.709": 6,
        "Gamma-2.5": 6,
    }
