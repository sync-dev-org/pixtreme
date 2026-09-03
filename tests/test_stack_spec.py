"""Specification tests for stack."""

from __future__ import annotations

import inspect
import math
from typing import Any

import numpy as np
import pytest

import pixtreme as px


def _frame(
    values: Any,
    *,
    colorspace: str = "sRGB",
    gamma: str = "linear",
    channels: str | tuple[str, ...] = "RGB",
    dtype: Any = np.float32,
) -> px.core.Frame:
    import cupy as cp

    return px.io.from_array(
        cp.asarray(np.asarray(values, dtype=dtype)),
        colorspace=colorspace,
        gamma=gamma,
        channels=channels,
    )


def _host(frame: px.core.Frame) -> np.ndarray:
    return px.io.to_array(
        frame,
    ).get()


def _assert_actionable(error: pytest.ExceptionInfo[ValueError]) -> None:
    message = str(error.value)
    assert message.index("why=") < message.index("what=") < message.index("how=")


def test_stack_images_public_signature_empty_input_and_single_input_copy() -> None:
    """v1-warp-affine acceptance 17: stack stays beside resize and warp_affine in transform."""
    signature = inspect.signature(px.transform.stack)
    assert tuple(signature.parameters) == ("images", "direction", "adapt")
    assert signature.parameters["images"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert signature.parameters["direction"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["direction"].default == "vertical"
    assert signature.parameters["adapt"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["adapt"].default is False
    assert "stack" in px.transform.__all__
    assert len(px.transform.__all__) == 3

    with pytest.raises(ValueError, match="empty|one or more"):
        px.transform.stack([])

    source = _frame([[[1.0], [2.0]]], colorspace="ACEScg", gamma="linear", channels=("Y",))
    for adapt in (False, True):
        result = px.transform.stack([source], adapt=adapt)
        assert isinstance(result, px.core.Frame)
        assert result is not source
        assert result.data.data.ptr != source.data.data.ptr
        assert (result.colorspace, result.gamma, result.channels) == (
            source.colorspace,
            source.gamma,
            source.channels,
        )
        np.testing.assert_array_equal(_host(result), _host(source))


@pytest.mark.parametrize(
    ("direction", "first_values", "second_values", "expected_shape"),
    (
        (
            "vertical",
            [[[1.0], [2.0]]],
            [[[3.0], [4.0]], [[5.0], [6.0]]],
            (3, 2, 1),
        ),
        (
            "horizontal",
            [[[1.0]], [[2.0]]],
            [[[3.0], [4.0]], [[5.0], [6.0]]],
            (2, 3, 1),
        ),
    ),
)
def test_stack_images_places_exact_copies_in_enumeration_order(
    direction: str,
    first_values: list[list[list[float]]],
    second_values: list[list[list[float]]],
    expected_shape: tuple[int, int, int],
) -> None:
    """v1-stack acceptance 2: vertical and horizontal stacking map input pixels exactly in order."""
    axis = 0 if direction == "vertical" else 1
    first = _frame(first_values, channels=("Y",))
    second = _frame(second_values, channels=("Y",))

    result = px.transform.stack([first, second], direction=direction)

    expected = np.concatenate((np.asarray(first_values), np.asarray(second_values)), axis=axis).astype(np.float32)
    assert result.shape == expected_shape
    np.testing.assert_array_equal(_host(result), expected)


@pytest.mark.parametrize("dtype", (np.float32, np.float16, np.uint8, np.uint16, np.uint32))
def test_default_stack_accepts_every_frame_storage_dtype_without_numeric_conversion(dtype: Any) -> None:
    """REQ-ARCH-005 and v1-stack acceptance 2: default stacking bit-copies every Frame storage dtype."""
    first = _frame([[[1], [2]]], channels=("Y",), dtype=dtype)
    second = _frame([[[3], [4]]], channels=("Y",), dtype=dtype)

    result = px.transform.stack((first, second))

    assert result.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(_host(result), np.asarray([[[1], [2]], [[3], [4]]], dtype=dtype))


def test_default_stack_preserves_uint32_identities_above_float32_exact_range() -> None:
    """REQ-ARCH-005: uint32 stacking is bit-preserving for identities above 2^24 that
    float32 cannot represent; a lossy uint32 -> float32 -> uint32 path fails here."""
    above_exact = np.uint32(16_777_217)  # 2^24 + 1: rounds to 16_777_216 through float32
    top = np.uint32(4_294_967_295)  # 0xffffffff: uint32 maximum
    first = _frame([[[above_exact], [top]]], channels=("Y",), dtype=np.uint32)
    second = _frame([[[np.uint32(16_777_219)], [np.uint32(0)]]], channels=("Y",), dtype=np.uint32)

    result = px.transform.stack((first, second))

    assert result.dtype == np.dtype(np.uint32)
    expected = np.asarray([[[16_777_217], [4_294_967_295]], [[16_777_219], [0]]], dtype=np.uint32)
    np.testing.assert_array_equal(_host(result), expected)


@pytest.mark.parametrize("direction", ("diagonal", "row", "vertical\t", ""))
def test_stack_direction_is_closed_vocabulary(direction: str) -> None:
    """v1-stack acceptance 2; v1-token-vocabulary acceptance 7: direction accepts only its two-token family."""
    source = _frame([[[0.0]]], channels=("Y",))

    with pytest.raises(ValueError) as error:
        px.transform.stack([source], direction=direction)

    message = str(error.value)
    assert repr(direction) in message
    assert "vertical" in message
    assert "horizontal" in message


@pytest.mark.parametrize(
    ("direction", "field", "first_value", "second_value"),
    (
        ("vertical", "width", 2, 3),
        ("horizontal", "height", 2, 3),
        ("vertical", "channels", ("R", "G", "B"), ("B", "G", "R")),
        ("vertical", "colorspace", "sRGB", "Rec.709"),
        ("vertical", "gamma", "linear", "sRGB"),
        ("vertical", "dtype", "float32", "float16"),
    ),
)
def test_default_stack_rejects_every_incompatible_axis_with_both_values(
    direction: str,
    field: str,
    first_value: object,
    second_value: object,
) -> None:
    """v1-stack acceptance 3: default stacking reports each incompatible axis and the conflicting values."""
    first_shape = (2, 2, 3)
    second_shape = (2, 2, 3)
    first_kwargs: dict[str, object] = {}
    second_kwargs: dict[str, object] = {}
    if field == "width":
        second_shape = (2, 3, 3)
    elif field == "height":
        second_shape = (3, 2, 3)
    elif field == "channels":
        first_kwargs["channels"] = first_value
        second_kwargs["channels"] = second_value
    elif field == "colorspace":
        first_kwargs["colorspace"] = first_value
        second_kwargs["colorspace"] = second_value
    elif field == "gamma":
        first_kwargs["gamma"] = first_value
        second_kwargs["gamma"] = second_value
    elif field == "dtype":
        first_kwargs["dtype"] = np.float32
        second_kwargs["dtype"] = np.float16

    first = _frame(np.zeros(first_shape), **first_kwargs)
    second = _frame(np.zeros(second_shape), **second_kwargs)

    with pytest.raises(ValueError) as error:
        px.transform.stack([first, second], direction=direction)

    message = str(error.value)
    assert field in message
    assert str(first_value) in message
    assert str(second_value) in message


@pytest.mark.parametrize(
    ("direction", "master_shape", "source_shape", "resized_shape", "expected_shape"),
    (
        ("vertical", (1, 3, 1), (1, 2, 1), (2, 3, 1), (3, 3, 1)),
        ("horizontal", (3, 1, 1), (2, 1, 1), (3, 2, 1), (3, 3, 1)),
    ),
)
def test_adapt_resizes_the_orthogonal_axis_to_first_with_half_up_aspect_rounding(
    direction: str,
    master_shape: tuple[int, int, int],
    source_shape: tuple[int, int, int],
    resized_shape: tuple[int, int, int],
    expected_shape: tuple[int, int, int],
) -> None:
    """v1-stack acceptance 4: adapt uses the first geometry and half-up aspect-preserving rounding."""
    axis = 0 if direction == "vertical" else 1
    master = _frame(np.arange(math.prod(master_shape)).reshape(master_shape), channels=("Y",))
    source = _frame(np.arange(math.prod(source_shape)).reshape(source_shape) + 10.0, channels=("Y",))
    resize_kwargs = (
        {"width": master.width, "height": resized_shape[0]}
        if direction == "vertical"
        else {"width": resized_shape[1], "height": master.height}
    )
    expected_source = px.transform.resize(source, **resize_kwargs)

    result = px.transform.stack([master, source], direction=direction, adapt=True)

    assert expected_source.shape == resized_shape
    assert result.shape == expected_shape
    expected = np.concatenate((_host(master), _host(expected_source)), axis=axis)
    np.testing.assert_array_equal(_host(result), expected)


def test_adapt_matches_channel_color_channel_then_default_resize_composition() -> None:
    """v1-stack acceptance 4 and 6: channel/color adaptation precedes geometry and inherits first metadata."""
    master = _frame(
        np.asarray(
            [
                [[0.02, 0.08, 0.20], [0.10, 0.30, 0.70], [0.80, 0.40, 0.05]],
                [[0.90, 0.20, 0.10], [0.03, 0.50, 0.12], [0.25, 0.75, 1.20]],
            ],
            dtype=np.float32,
        ),
        colorspace="ACEScg",
        gamma="linear",
    )
    source = _frame(
        np.asarray([[[0.30, 0.90, 0.02], [0.10, 0.04, 0.80]]], dtype=np.float32),
        channels=("Cb", "Cr", "Y"),
        colorspace="Rec.709",
        gamma="Rec.709",
    )
    complete = px.color.ycbcr_to_rgb(source)
    transformed = px.color.rgb_to_rgb(
        complete,
        output_colorspace=master.colorspace,
        output_gamma=master.gamma,
    )
    ordered = px.channel.shuffle(**{label: (transformed, label) for label in master.channels})
    resized = px.transform.resize(ordered, width=master.width, height=2)
    expected = np.concatenate((_host(master), _host(resized)), axis=0)

    result = px.transform.stack([master, source], adapt=True)

    assert result.shape == (4, 3, 3)
    assert (result.colorspace, result.gamma, result.channels) == (
        master.colorspace,
        master.gamma,
        master.channels,
    )
    np.testing.assert_array_equal(_host(result), expected)


@pytest.mark.parametrize(
    (
        "master_channels",
        "source_channels",
        "master_colorspace",
        "source_colorspace",
        "master_gamma",
        "source_gamma",
    ),
    (
        (("Cr", "A", "Y", "Cb"), ("A", "B", "R", "G"), "Rec.2020", "sRGB", "linear", "sRGB"),
        (("B", "R", "G"), ("Cb", "Cr", "Y"), "ACEScg", "Rec.709", "linear", "Rec.709"),
    ),
)
def test_adapt_matches_public_channel_color_channel_composition_bit_exactly(
    master_channels: tuple[str, ...],
    source_channels: tuple[str, ...],
    master_colorspace: str,
    source_colorspace: str,
    master_gamma: str,
    source_gamma: str,
) -> None:
    """v1-stack acceptance 4: RGB/YCbCr adaptation follows the deterministic public-op composition."""
    master = _frame(
        np.zeros((1, 2, len(master_channels)), dtype=np.float32),
        channels=master_channels,
        colorspace=master_colorspace,
        gamma=master_gamma,
    )
    source_values = np.arange(1, 1 + 2 * len(source_channels), dtype=np.float32).reshape(1, 2, -1) / 10.0
    source = _frame(
        source_values,
        channels=source_channels,
        colorspace=source_colorspace,
        gamma=source_gamma,
    )
    complete = px.color.ycbcr_to_rgb(source) if {"Y", "Cb", "Cr"} <= set(source.channels) else source
    transformed = px.color.rgb_to_rgb(
        complete,
        output_colorspace=master.colorspace,
        output_gamma=master.gamma,
    )
    converted = (
        px.color.rgb_to_ycbcr(transformed, matrix=master.matrix)
        if {"Y", "Cb", "Cr"} <= set(master.channels)
        else transformed
    )
    expected_source = px.channel.shuffle(**{label: (converted, label) for label in master.channels})

    result = px.transform.stack([master, source], adapt=True)
    actual_source = _host(result)[master.height :]

    assert result.channels == master.channels
    np.testing.assert_array_equal(actual_source, _host(expected_source))
    if "A" in source_channels:
        np.testing.assert_array_equal(
            actual_source[..., master.channels.index("A")],
            source_values[..., source_channels.index("A")],
        )


def test_adapt_reorders_an_equal_arbitrary_channel_set_without_changing_values() -> None:
    """v1-stack acceptance 4: equal channel-label sets follow the first ordering without color conversion."""
    master = _frame(np.zeros((1, 2, 3), dtype=np.float32), channels=("matte", "Z", "A"))
    source = _frame(
        np.asarray([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]], dtype=np.float32),
        channels=("A", "Z", "matte"),
    )
    expected_source = px.channel.shuffle(**{label: (source, label) for label in master.channels})

    result = px.transform.stack([master, source], adapt=True)

    np.testing.assert_array_equal(_host(result)[master.height :], _host(expected_source))


@pytest.mark.parametrize(
    ("master_channels", "source_channels"),
    (
        (("R", "G", "B"), ("foo", "bar", "baz")),
        (("R", "G", "B"), ("R", "G")),
        (("R", "G", "B"), ("Y",)),
        (("R", "G", "B", "A"), ("Y", "Cb", "Cr", "Z")),
    ),
)
def test_adapt_rejects_channel_pairs_without_a_deterministic_conversion(
    master_channels: tuple[str, ...],
    source_channels: tuple[str, ...],
) -> None:
    """v1-stack acceptance 5: adapt rejects unknown, unequal-count, matte-promotion, and alpha-mismatch pairs."""
    master = _frame(np.zeros((1, 1, len(master_channels)), dtype=np.float32), channels=master_channels)
    source = _frame(np.zeros((1, 1, len(source_channels)), dtype=np.float32), channels=source_channels)

    with pytest.raises(ValueError) as error:
        px.transform.stack([master, source], adapt=True)

    _assert_actionable(error)
    message = str(error.value)
    assert "channels" in message
    assert repr(master_channels) in message
    assert repr(source_channels) in message


def test_adapt_preserves_dtype_and_color_conversion_fail_fast_boundaries() -> None:
    """REQ-API-012 / v1-stack acceptance 5: adaptation failures are actionable through their causes."""
    rgb = np.zeros((1, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="dtype.*float32.*float16"):
        px.transform.stack([_frame(rgb), _frame(rgb, dtype=np.float16)], adapt=True)

    with pytest.raises(ValueError) as error:
        px.transform.stack([_frame(rgb, dtype=np.float16), _frame(rgb, dtype=np.float16)], adapt=True)
    assert "float32" in str(error.value) and "cast_dtype" in str(error.value)

    uint = np.zeros((1, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError) as error:
        px.transform.stack([_frame(uint, dtype=np.uint8), _frame(uint, dtype=np.uint8)], adapt=True)
    message = str(error.value)
    assert message.index("recode_dtype") < message.index("dequantize")

    with pytest.raises(ValueError) as error:
        px.transform.stack(
            [
                _frame(rgb, colorspace="ACEScg", channels=("Y", "Cb", "Cr")),
                _frame(rgb, colorspace="ACEScg", channels=("R", "G", "B")),
            ],
            adapt=True,
        )
    _assert_actionable(error)
    assert "deterministic" in str(error.value) and "matrix" in str(error.value)
    cause = error.value.__cause__
    assert cause is not None
    cause_message = str(cause)
    assert "why=" in cause_message
    assert "; what=" in cause_message
    assert "; how=" in cause_message

    y = np.zeros((1, 2, 1), dtype=np.float32)
    with pytest.raises(ValueError) as error:
        px.transform.stack(
            [
                _frame(y, gamma="linear", channels=("Y",)),
                _frame(y, gamma="sRGB", channels=("Y",)),
            ],
            adapt=True,
        )
    _assert_actionable(error)
    assert "channels" in str(error.value) and "gamma" in str(error.value)


def test_stack_images_rejects_non_frame_members() -> None:
    """v1-stack acceptance 1: every collection member must be a Frame."""
    with pytest.raises(ValueError, match="Frame"):
        px.transform.stack([object()])
