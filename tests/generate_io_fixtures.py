"""Reproducible OpenEXR fixtures for the v1-io specification tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def write_exr(
    path: Path,
    channels: Mapping[str, NDArray[np.generic]],
    *,
    header: Mapping[str, object] | None = None,
) -> None:
    """Write one-part EXR, including Blender-style dotted channel names."""
    from openexr_dev_oracle import OpenEXR

    OpenEXR.File(dict(header or {}), dict(channels)).write(str(path))


def write_multipart_exr(
    path: Path,
    parts: Sequence[tuple[str, Mapping[str, NDArray[np.generic]], Mapping[str, object]]],
) -> None:
    """Write a true multi-part EXR with explicit part names."""
    from openexr_dev_oracle import OpenEXR

    dimensions = [next(iter(channels.values())).shape for _, channels, _ in parts]
    display_height = max(shape[0] for shape in dimensions)
    display_width = max(shape[1] for shape in dimensions)
    display_window = (
        np.array((0, 0), dtype=np.int32),
        np.array((display_width - 1, display_height - 1), dtype=np.int32),
    )
    exr_parts = []
    for name, channels, header in parts:
        part_header = {**header, "displayWindow": display_window}
        exr_parts.append(OpenEXR.Part(part_header, dict(channels), name))
    OpenEXR.File(exr_parts).write(str(path))
