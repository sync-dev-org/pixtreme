"""Cross-module float32 input contracts for public processing operations."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

import pixtreme as px

_NON_FLOAT32_DTYPES = (np.float16, np.uint8, np.uint16, np.uint32)
_OPERATION_KWARGS: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "draw.line",
        {"start": (0.0, 0.0), "end": (1.0, 1.0), "color": (0.0, 0.0, 0.0), "thickness": 1.0, "opacity": 0.0},
    ),
    (
        "draw.polyline",
        {
            "points": ((0.0, 0.0), (1.0, 1.0)),
            "color": (0.0, 0.0, 0.0),
            "thickness": 1.0,
            "opacity": 0.0,
        },
    ),
    (
        "draw.rectangle",
        {
            "top_left": (0.0, 0.0),
            "bottom_right": (1.0, 1.0),
            "color": (0.0, 0.0, 0.0),
            "fill": True,
            "opacity": 0.0,
        },
    ),
    (
        "draw.circle",
        {"center": (0.5, 0.5), "radius": 0.5, "color": (0.0, 0.0, 0.0), "fill": True, "opacity": 0.0},
    ),
    (
        "draw.ellipse",
        {
            "center": (0.5, 0.5),
            "radii": (0.5, 0.25),
            "color": (0.0, 0.0, 0.0),
            "fill": True,
            "opacity": 0.0,
        },
    ),
    (
        "draw.polygon",
        {"points": ((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)), "color": (0.0, 0.0, 0.0), "opacity": 0.0},
    ),
    (
        "draw.text",
        {"text": "", "position": (0.0, 0.0), "size": 12.0, "color": (0.0, 0.0, 0.0)},
    ),
    ("filter.gaussian_blur", {"sigma": 1.0}),
    ("filter.box_blur", {"size": 1}),
    ("filter.median_blur", {"size": 1}),
    ("filter.bilateral_blur", {"sigma_space": 1.0, "sigma_value": 1.0}),
    ("filter.directional_blur", {"angle": 0.0, "length": 1.0}),
    ("filter.zoom_blur", {"amount": 1.0}),
    ("filter.spin_blur", {"angle": 1.0}),
    ("filter.vector_blur", {}),
    ("filter.lens_blur", {"radius": 0.0}),
    ("filter.convolve_box", {"size": 1, "normalize": True}),
    ("transform.resize", {"width": 2, "height": 2}),
)


def _frame(
    dtype: type[np.generic],
    *,
    channels: str | Sequence[str] = "RGB",
) -> px.core.Frame:
    import cupy as cp

    channel_count = len(channels)
    return px.io.from_array(
        cp.zeros((2, 2, channel_count), dtype=dtype),
        colorspace="ACEScg",
        gamma="linear",
        channels=channels,
    )


def _public_operation(path: str) -> Any:
    module_name, operation_name = path.split(".")
    return getattr(getattr(px, module_name), operation_name)


def _assert_float32_guidance(
    error: pytest.ExceptionInfo[ValueError], *, operation: str, dtype: type[np.generic]
) -> None:
    message = str(error.value)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message
    assert operation in message
    assert "float32" in message
    assert np.dtype(dtype).name in message

    guidance_paths = re.findall(r"px\.([a-z_]+)\.([a-z_]+)", message)
    assert guidance_paths
    assert any(module_name == "values" for module_name, _ in guidance_paths)
    for module_name, operation_name in guidance_paths:
        assert hasattr(getattr(px, module_name), operation_name)


@pytest.mark.parametrize(("operation", "kwargs"), _OPERATION_KWARGS)
@pytest.mark.parametrize("dtype", _NON_FLOAT32_DTYPES)
def test_public_processing_operations_reject_non_float32_before_identity_shortcuts(
    operation: str,
    kwargs: dict[str, Any],
    dtype: type[np.generic],
) -> None:
    """REQ-ARCH-005: every processing entry fails before its identity shortcut and names a public cast route."""
    call_kwargs = dict(kwargs)
    if operation == "filter.vector_blur":
        call_kwargs["vector"] = _frame(np.float32, channels=("x", "y"))

    with pytest.raises(ValueError) as error:
        _public_operation(operation)(_frame(dtype), **call_kwargs)

    _assert_float32_guidance(error, operation=operation, dtype=dtype)


@pytest.mark.parametrize("dtype", _NON_FLOAT32_DTYPES)
def test_vector_blur_rejects_non_float32_vector_frame(dtype: type[np.generic]) -> None:
    """REQ-ARCH-005: vector_blur validates the vector Frame's storage dtype as well as the image dtype."""
    with pytest.raises(ValueError) as error:
        px.filter.vector_blur(_frame(np.float32), vector=_frame(dtype, channels=("x", "y")))

    _assert_float32_guidance(error, operation="filter.vector_blur", dtype=dtype)


@pytest.mark.parametrize("dtype", _NON_FLOAT32_DTYPES)
def test_text_rejects_non_float32_before_opacity_identity(dtype: type[np.generic]) -> None:
    """REQ-ARCH-005: text validates storage before the opacity-zero identity path."""
    with pytest.raises(ValueError) as error:
        px.draw.text(
            _frame(dtype),
            text="visible",
            position=(0.0, 0.0),
            size=12.0,
            color=(0.0, 0.0, 0.0),
            opacity=0.0,
        )

    _assert_float32_guidance(error, operation="draw.text", dtype=dtype)
