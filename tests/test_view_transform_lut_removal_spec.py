"""Acceptance tests for removing the private view-transform LUT path."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, get_args

import cupy as cp
import numpy as np
import pytest

import pixtreme as px

ROOT = Path(__file__).resolve().parents[1]

_SUPPORTED_COMBINATIONS = (
    ("ACES-1.3", "Rec.709", "BT.1886"),
    ("ACES-1.3", "sRGB", "sRGB"),
    ("ACES-2.0", "Rec.709", "BT.1886"),
    ("ACES-2.0", "sRGB", "sRGB"),
    ("BT.2408", "Rec.2020", "HLG"),
    ("BT.2408", "Rec.2020", "PQ"),
)
_RETAINED_OUTPUT_SHA256 = {
    ("ACES-1.3", "Rec.709", "BT.1886"): "9bfc7d64433e8d7627624fc14e0c3f62872ecb6d215f54b7ba13189fe1eddfbc",
    ("ACES-1.3", "sRGB", "sRGB"): "2a30b0ea3b00735f677123ff547ea27bbd00e2abaafdebe6862201a8f2f43529",
    ("ACES-2.0", "Rec.709", "BT.1886"): "d173d5ca018063694d5b36d209ae961b5c3ec5c86d3b3f05b198895ae6ac7162",
    ("ACES-2.0", "sRGB", "sRGB"): "37446c175bc687fc14d47e658e39a54e93f519340715b2fa61e788a59f896e30",
    ("BT.2408", "Rec.2020", "HLG"): "128d87c04d8a72ed376a02aa1ac2a8d065b02d65ad782dd274bcaf17533a7f88",
    ("BT.2408", "Rec.2020", "PQ"): "cbb12529469b6f53cbd8ce0d522de684d91cee9073beb3ec7c2644497fee5dae",
}


def _frame(values: Any) -> px.core.Frame:
    data = cp.asarray(values, dtype=cp.float32).reshape(-1, 1, 3)
    return px.io.from_array(data, colorspace="ACEScg", gamma="linear", channels="RGB")


@pytest.mark.parametrize("retired_token", ("aces-1.3-lut", "aces-2.0-lut"))
def test_retired_lut_tonemap_tokens_fail_with_the_closed_set_recipe(retired_token: str) -> None:
    """v1-view-transform-lut-removal acceptance 1 and 6: retired tokens fail before pixel processing."""
    with pytest.raises(ValueError) as error:
        px.color.rgb_to_rgb(
            _frame((0.18, 0.18, 0.18)),
            output_colorspace="sRGB",
            output_gamma="sRGB",
            tonemap=retired_token,
        )

    message = str(error.value)
    assert all(part in message for part in ("why=", "what=", "how="))
    assert retired_token in message
    how = message.split("how=", maxsplit=1)[1]
    assert "ACES-1.3" in how
    assert "ACES-2.0" in how
    assert "BT.2408" in how
    assert "-lut" not in how
    assert np.isfinite(_frame((0.18, 0.18, 0.18)).data.get()).all()


def test_public_vocabulary_and_runtime_supply_exactly_three_tokens_and_six_combinations() -> None:
    """v1-view-transform-lut-removal acceptance 2, 4, and 8: public and runtime closed sets agree."""
    from pixtreme._color.transform import _SUPPORTED_COMBINATIONS as runtime_combinations

    assert get_args(px.core.Tonemap) == ("ACES-1.3", "ACES-2.0", "BT.2408")
    assert runtime_combinations == _SUPPORTED_COMBINATIONS


def test_characterization_retained_tonemap_outputs_freeze_pre_removal_float32_bits() -> None:
    """v1-view-transform-lut-removal acceptance 2: characterization of retained-route output bits.

    Characterization test: the SHA-256 digests freeze the float32 output bits that the pre-removal
    implementation produced for one representative input on the six retained tonemap routes. Exact
    bits are not an externally specified contract, and rendered-value correctness is covered
    independently by the OCIO oracle corpus suites; the digests exist to detect unintended numeric
    drift while the removal is integrated. Regenerate or retire the digests when an intentional
    numeric change to the analytic renderers or the BT.2408 mapping is accepted, or when the
    executing GPU / toolchain combination changes the produced bits.
    """
    values = np.asarray(
        (
            ((-0.125, 0.0, 0.18), (0.18, 0.5, 1.0), (1.5, 0.25, 0.05), (4.0, 2.0, 0.5)),
            ((0.01, 0.02, 0.03), (0.9, 0.1, 0.4), (0.0, 1.0, 0.0), (16.0, 8.0, 2.0)),
        ),
        dtype=np.float32,
    )
    source = px.io.from_array(cp.asarray(values), colorspace="ACEScg", gamma="linear", channels="RGB")

    for combination, expected_digest in _RETAINED_OUTPUT_SHA256.items():
        tonemap, output_colorspace, output_gamma = combination
        result = px.color.rgb_to_rgb(
            source,
            output_colorspace=output_colorspace,
            output_gamma=output_gamma,
            tonemap=tonemap,
        )
        assert hashlib.sha256(result.data.get().tobytes()).hexdigest() == expected_digest


def test_private_runtime_module_and_packaged_archives_are_absent() -> None:
    """v1-view-transform-lut-removal acceptance 3: private evaluation code and package archives are removed."""
    assert not (ROOT / "src" / "pixtreme" / "_color" / "view_transform.py").exists()
    assert not tuple((ROOT / "src" / "pixtreme" / "data").glob("view_transform_*.npz"))
