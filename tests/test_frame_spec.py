"""Specification tests for the public Frame behavior."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import pixtreme as px


def _sample_data(*, channel_count: int = 3, dtype: str = "float32") -> Any:
    import cupy as cp

    return cp.arange(2 * 3 * channel_count, dtype=dtype).reshape(2, 3, channel_count)


def _assert_actionable(error: pytest.ExceptionInfo[ValueError]) -> None:
    message = str(error.value)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message


@pytest.mark.parametrize("value", ("", "rgb", 42, ["R", ""]))
def test_channel_validation_errors_are_actionable(value: object) -> None:
    """REQ-API-012: every compact/sequence channel rejection carries why, what, and recovery guidance."""
    with pytest.raises(ValueError) as error:
        px.core.channels(value)  # type: ignore[arg-type]

    _assert_actionable(error)


@pytest.mark.parametrize(
    "metadata",
    (
        {"colorspace": "unknown", "gamma": "linear", "matrix": None},
        {"colorspace": "sRGB", "gamma": "unknown", "matrix": None},
        {"colorspace": "sRGB", "gamma": "linear", "matrix": "unknown"},
    ),
)
def test_frame_metadata_token_errors_are_actionable(metadata: dict[str, str | None]) -> None:
    """REQ-API-012: metadata token rejection identifies the axis, received token, and accepted recovery values."""
    with pytest.raises(ValueError) as error:
        px.io.from_array(_sample_data(), channels="RGB", **metadata)  # type: ignore[arg-type]

    _assert_actionable(error)


def test_frame_data_validation_errors_are_actionable() -> None:
    """REQ-API-012: Frame data type, rank, dtype, and channel-count rejections use the fixed-order three-element contract."""
    import cupy as cp

    class ValidationProbe:
        data: dict[str, object] = {}

    with pytest.raises(ValueError) as non_cupy_error:
        px.core.Frame._validate_data(np.zeros((1, 1, 3), dtype=np.float32), ValidationProbe())  # type: ignore[arg-type]
    _assert_actionable(non_cupy_error)

    invalid_construction_data = (
        cp.zeros((1, 3), dtype=cp.float32),
        cp.zeros((1, 1, 3), dtype=cp.float64),
    )
    for data in invalid_construction_data:
        with pytest.raises(ValueError) as error:
            px.core.Frame(data=data, colorspace="sRGB", gamma="linear", channels="RGB")
        _assert_actionable(error)

    frame = px.io.from_array(_sample_data(), colorspace="sRGB", gamma="linear", channels="RGB")
    with pytest.raises(ValueError) as data_channels_error:
        frame.data = cp.zeros((1, 1, 4), dtype=cp.float32)
    _assert_actionable(data_channels_error)

    with pytest.raises(ValueError) as channels_data_error:
        frame.channels = "RGBA"
    _assert_actionable(channels_data_error)


@pytest.mark.parametrize(
    "shape",
    ((0, 2, 3), (2, 0, 3), (2, 2, 0)),
    ids=("zero-height", "zero-width", "zero-channels"),
)
def test_frame_rejects_empty_hwc_dimensions_at_construction_and_assignment(
    shape: tuple[int, int, int],
) -> None:
    """REQ-ARCH-002: Frame construction and data assignment require H, W, and C to each be at least 1."""
    import cupy as cp

    labels = tuple(f"channel-{index}" for index in range(shape[2]))
    with pytest.raises(ValueError) as construction_error:
        px.core.Frame(
            data=cp.empty(shape, dtype=cp.float32),
            colorspace="sRGB",
            gamma="linear",
            channels=labels,
        )
    _assert_actionable(construction_error)
    assert repr(shape) in str(construction_error.value)
    assert "at least 1" in str(construction_error.value)

    frame = px.io.from_array(_sample_data(), colorspace="sRGB", gamma="linear", channels="RGB")
    original = frame.data
    with pytest.raises(ValueError) as assignment_error:
        frame.data = cp.empty(shape, dtype=cp.float32)
    _assert_actionable(assignment_error)
    assert repr(shape) in str(assignment_error.value)
    assert "at least 1" in str(assignment_error.value)
    assert frame.data is original


def test_from_array_constructs_from_hwc_cupy_and_checks_channel_count_and_rank() -> None:
    """v1-boundary-api acceptance 1 and 3: from_array accepts structurally consistent HWC CuPy data."""
    import cupy as cp

    data = _sample_data()
    result = px.io.from_array(data, colorspace="sRGB", gamma="srgb", channels="RGB")

    assert isinstance(result, px.core.Frame)
    assert result.data is data

    with pytest.raises(ValueError, match="channel"):
        px.io.from_array(data, colorspace="sRGB", gamma="srgb", channels="RGBA")
    with pytest.raises(ValueError, match="HWC"):
        px.io.from_array(cp.zeros((2, 3), dtype=cp.float32), colorspace="sRGB", gamma="srgb", channels="RGB")
    with pytest.raises(ValueError, match="HWC"):
        px.io.from_array(cp.zeros((1, 2, 3, 4), dtype=cp.float32), colorspace="sRGB", gamma="srgb", channels="RGBA")


def test_from_array_makes_non_contiguous_input_c_contiguous() -> None:
    """v1-boundary-api acceptance 4: from_array copies non-contiguous input under copy=None."""
    import cupy as cp

    source = cp.arange(4 * 6 * 3, dtype=cp.float32).reshape(4, 6, 3)[:, ::2, :]
    expected = cp.asnumpy(source)
    assert not source.flags.c_contiguous

    result = px.io.from_array(source, colorspace="sRGB", gamma="srgb", channels="RGB")

    assert result.data.flags.c_contiguous
    assert result.data.data.ptr != source.data.ptr
    np.testing.assert_array_equal(
        px.io.to_array(
            result,
        ).get(),
        expected,
    )


def test_frame_accepts_only_the_specified_storage_dtypes() -> None:
    """v1-exr-runtime-independence acceptance 7: Frame storage accepts all five public storage dtypes."""
    import cupy as cp

    for dtype in (cp.float32, cp.float16, cp.uint8, cp.uint16, cp.uint32):
        data = cp.zeros((1, 2, 3), dtype=dtype)
        assert px.io.from_array(data, colorspace="sRGB", gamma="srgb", channels="RGB").dtype == data.dtype

    for dtype in (cp.float64, cp.int32, cp.int8, cp.bool_):
        with pytest.raises(ValueError, match="dtype"):
            px.io.from_array(cp.zeros((1, 2, 3), dtype=dtype), colorspace="sRGB", gamma="srgb", channels="RGB")


@pytest.mark.parametrize(
    "colorspace",
    ("sRGB", "Rec.709", "Rec.2020", "ACES2065-1", "ACEScg", "S-Gamut3", "S-Gamut3.Cine"),
)
def test_frame_accepts_each_colorspace_token(colorspace: str) -> None:
    """v1-frame-core acceptance 7: each canonical colorspace token is accepted exactly as written."""
    result = px.io.from_array(_sample_data(), colorspace=colorspace, gamma="linear", channels=["R", "G", "B"])
    assert result.colorspace == colorspace


@pytest.mark.parametrize(
    "gamma",
    ("linear", "srgb", "rec709", "bt1886", "pq", "hlg", "s-log3", "logc4", "cineon", "2.2", "2.4", "2.6"),
)
def test_frame_accepts_each_gamma_token(gamma: str) -> None:
    """v1-color-semantics acceptance 27: Frame accepts the complete gamma vocabulary including 2.6."""
    result = px.io.from_array(_sample_data(), colorspace="sRGB", gamma=gamma, channels=["red", "green", "blue"])
    assert result.gamma == gamma
    assert result.channels == ("red", "green", "blue")


def test_frame_rejects_unknown_or_wrong_case_metadata_tokens() -> None:
    """v1-frame-core acceptance 7: unknown colorspace/gamma tokens fail fast and matching is case-sensitive."""
    data = _sample_data()

    for colorspace in ("srgb", "ACESCG", "unknown"):
        with pytest.raises(ValueError, match="colorspace"):
            px.io.from_array(data, colorspace=colorspace, gamma="linear", channels="RGB")
    for gamma in ("Linear", "sRGB", "unknown"):
        with pytest.raises(ValueError, match="gamma"):
            px.io.from_array(data, colorspace="sRGB", gamma=gamma, channels="RGB")


def test_metadata_assignment_revalidates_transactionally_without_touching_data() -> None:
    """v1-frame-core acceptance 8: metadata assignment validates and a failed assignment preserves state."""
    result = px.io.from_array(_sample_data(), colorspace="sRGB", gamma="srgb", channels="RGB")
    data_pointer = result.data.data.ptr

    result.colorspace = "ACEScg"
    result.gamma = "linear"
    result.channels = "BGR"
    assert (result.colorspace, result.gamma, result.channels) == ("ACEScg", "linear", ("B", "G", "R"))

    with pytest.raises(ValueError, match="colorspace"):
        result.colorspace = "acescg"
    assert result.colorspace == "ACEScg"
    with pytest.raises(ValueError, match="gamma"):
        result.gamma = "Linear"
    assert result.gamma == "linear"
    with pytest.raises(ValueError, match="channel"):
        result.channels = "RGBA"
    assert result.channels == ("B", "G", "R")
    assert result.data.data.ptr == data_pointer


def test_frame_data_assignment_checks_channel_count_transactionally() -> None:
    """v1-channel-transform acceptance 14: data assignment preserves state when its channel count is invalid."""
    import cupy as cp

    source = px.io.from_array(_sample_data(), colorspace="sRGB", gamma="linear", channels="RGB")
    original = source.data
    invalid = cp.zeros((2, 2, 4), dtype=cp.float32)

    with pytest.raises(ValueError, match="channel"):
        source.data = invalid

    assert source.data is original
    assert source.channels == ("R", "G", "B")

    replacement = cp.ones((2, 2, 3), dtype=cp.float32)
    source.data = replacement
    assert source.data is replacement
    assert source.shape == (2, 2, 3)


def test_channels_normalizes_compact_and_sequence_inputs() -> None:
    """v1-frame-core acceptance 10: channels uses greedy known-label parsing and permits unknown sequence labels."""
    assert px.core.channels("YCbCrA") == ("Y", "Cb", "Cr", "A")
    assert px.core.channels("RGBA") == ("R", "G", "B", "A")
    assert px.core.channels(["R", "temperature", "Z"]) == ("R", "temperature", "Z")

    for value in ("rgb", "YCbcr", "temperature", ""):
        with pytest.raises(ValueError, match="channel"):
            px.core.channels(value)
    with pytest.raises(ValueError, match="channel"):
        px.core.channels(["R", ""])


def test_all_channel_entry_points_accept_the_same_input_forms() -> None:
    """v1-boundary-api acceptance 1 and 11: constructor, assignment, and exit share channel normalization."""
    result = px.io.from_array(_sample_data(), colorspace="sRGB", gamma="srgb", channels="RGB")
    assert result.channels == px.core.channels("RGB")

    result.channels = ["B", "G", "R"]
    assert result.channels == px.core.channels(["B", "G", "R"])
    output = px.io.to_array(result, channels="RGB").get()
    assert output.shape == result.shape


def test_channels_read_as_tuple_and_repr_uses_compact_form_only_for_known_labels() -> None:
    """v1-frame-core acceptance 11: channels are tuples and repr compacts only entirely known labels."""
    known = px.io.from_array(_sample_data(), colorspace="sRGB", gamma="srgb", channels="RGB")
    unknown = px.io.from_array(
        _sample_data(channel_count=2),
        colorspace="sRGB",
        gamma="linear",
        channels=["depth", "mask"],
    )

    assert known.channels == ("R", "G", "B")
    assert "channels='RGB'" in repr(known)
    assert "channels=('depth', 'mask')" in repr(unknown)


def test_frame_exposes_read_only_geometry_and_dtype_without_operator_forwarding() -> None:
    """v1-frame-core acceptance 15: read-only geometry/dtype properties do not make Frame array-like."""
    source = px.io.from_array(_sample_data(), colorspace="sRGB", gamma="srgb", channels="RGB")

    assert source.width == 3
    assert source.height == 2
    assert source.shape == (2, 3, 3)
    assert source.dtype == source.data.dtype
    for name, value in (("width", 10), ("height", 10), ("shape", (1, 1, 3)), ("dtype", "float16")):
        with pytest.raises((AttributeError, ValueError)):
            setattr(source, name, value)
    with pytest.raises(TypeError):
        _ = source + 1
