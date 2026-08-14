"""EXR metadata and channel-order preservation contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from generate_io_fixtures import write_exr
from numpy.typing import NDArray

import pixtreme as px

_ACES_CHROMATICITIES = (
    0.7347,
    0.2653,
    0.0,
    1.0,
    0.0001,
    -0.077,
    0.32168,
    0.33767,
)


def test_exr_metadata_priority_default_channels_explicit_order_and_header_are_preserved(tmp_path: Path) -> None:
    """v1-exr-runtime-independence acceptance 5: runtime independence preserves EXR metadata and channel order."""
    channels: dict[str, NDArray[np.float16]] = {
        "R": np.full((2, 3), 1.0, dtype=np.float16),
        "G": np.full((2, 3), 2.0, dtype=np.float16),
        "B": np.full((2, 3), 3.0, dtype=np.float16),
        "A": np.full((2, 3), 4.0, dtype=np.float16),
        "depth.Z": np.full((2, 3), 5.0, dtype=np.float16),
    }
    path = tmp_path / "metadata.exr"
    write_exr(
        path,
        channels,
        header={"chromaticities": _ACES_CHROMATICITIES, "acesImageContainerFlag": 1},
    )

    default = px.io.read_image(path)
    explicit = px.io.read_image(path, channels=("depth.Z", "B", "R"), colorspace="ACEScg", gamma="2.2")
    header = px.io.read_header(path)

    assert (default.channels, default.colorspace, default.gamma) == (("R", "G", "B", "A"), "ACES2065-1", "linear")
    assert (explicit.channels, explicit.colorspace, explicit.gamma) == (("depth.Z", "B", "R"), "ACEScg", "2.2")
    np.testing.assert_array_equal(
        px.io.to_array(explicit).get(),
        np.stack([channels["depth.Z"], channels["B"], channels["R"]], axis=2).astype(np.float32),
    )
    assert (header.color.colorspace, header.color.mappable) == ("ACES2065-1", True)
    assert set(header.color.raw) >= {"chromaticities", "acesImageContainerFlag"}
