"""Specification, numerical-oracle, and documentation tests for morphology filters."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

import pixtreme as px

MORPHOLOGY_NAMES = (
    "erosion",
    "dilation",
    "opening",
    "closing",
    "morphological_gradient",
    "white_tophat",
    "black_tophat",
)
BORDERS = ("mirror", "replicate", "wrap", "constant")
SHAPES = ("disk", "square")


def _frame(
    values: Any,
    *,
    colorspace: str = "sRGB",
    gamma: str = "linear",
    channels: str | list[str] = "RGB",
    dtype: str = "float32",
) -> px.core.Frame:
    import cupy as cp

    return px.io.from_array(
        cp.asarray(np.asarray(values, dtype=np.dtype(dtype))),
        colorspace=colorspace,
        gamma=gamma,
        channels=channels,
    )


def _assert_actionable(error: pytest.ExceptionInfo[ValueError]) -> None:
    message = str(error.value)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message


def _border_index(index: int, extent: int, border: str) -> int:
    if extent <= 1:
        return 0
    if border == "replicate":
        return min(max(index, 0), extent - 1)
    if border == "wrap":
        return index % extent
    period = 2 * extent - 2
    reflected = index % period
    return reflected if reflected < extent else period - reflected


def _primitive_reference(
    source: np.ndarray,
    *,
    radius: int,
    shape: str,
    border: str,
    border_value: float,
    dilate: bool,
) -> np.ndarray:
    """Independent scalar NumPy reference derived from v1-morphology acceptance 2, 4, and 7."""
    output = np.empty_like(source, dtype=np.float32)
    height, width, channel_count = source.shape
    for y in range(height):
        for x in range(width):
            for channel in range(channel_count):
                samples: list[np.float32] = []
                for offset_y in range(-radius, radius + 1):
                    for offset_x in range(-radius, radius + 1):
                        if shape == "disk" and offset_x * offset_x + offset_y * offset_y > radius * radius:
                            continue
                        sample_x = x + offset_x
                        sample_y = y + offset_y
                        if border == "constant" and not (0 <= sample_x < width and 0 <= sample_y < height):
                            samples.append(np.float32(border_value))
                        else:
                            source_x = _border_index(sample_x, width, border)
                            source_y = _border_index(sample_y, height, border)
                            samples.append(source[source_y, source_x, channel])
                output[y, x, channel] = max(samples) if dilate else min(samples)
    return output


def _compound_reference(
    source: np.ndarray,
    *,
    operation: str,
    radius: int,
    shape: str,
    border: str,
    border_value: float,
) -> np.ndarray:
    arguments = {"radius": radius, "shape": shape, "border": border, "border_value": border_value}
    eroded = _primitive_reference(source, dilate=False, **arguments)
    dilated = _primitive_reference(source, dilate=True, **arguments)
    if operation == "opening":
        return _primitive_reference(eroded, dilate=True, **arguments)
    if operation == "closing":
        return _primitive_reference(dilated, dilate=False, **arguments)
    if operation == "morphological_gradient":
        return dilated - eroded
    if operation == "white_tophat":
        opened = _primitive_reference(eroded, dilate=True, **arguments)
        return source - opened
    closed = _primitive_reference(dilated, dilate=False, **arguments)
    return closed - source


def test_morphology_public_signatures_are_exact_frame_only_contracts() -> None:
    """v1-morphology acceptance 1: seven exact keyword APIs require radius and return Frame values."""
    import cupy as cp

    source = _frame(np.arange(9, dtype=np.float32).reshape(3, 3, 1), channels=["matte"])
    for name in MORPHOLOGY_NAMES:
        function = getattr(px.morphology, name)
        signature = inspect.signature(function)
        assert tuple(signature.parameters) == ("frame", "radius", "shape", "border", "border_value")
        for parameter in ("radius", "shape", "border", "border_value"):
            assert signature.parameters[parameter].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters["radius"].default is inspect.Parameter.empty
        assert signature.parameters["shape"].default == "disk"
        assert signature.parameters["border"].default == "replicate"
        assert signature.parameters["border_value"].default is None
        assert name in px.morphology.__all__
        assert isinstance(function(source, radius=1), px.core.Frame)
        with pytest.raises(ValueError) as error:
            function(cp.zeros((3, 3, 1), dtype=cp.float32), radius=1)
        _assert_actionable(error)


@pytest.mark.parametrize("radius", (0, -1, 1.0, True, np.int64(1), "1"))
def test_morphology_radius_is_a_built_in_int_of_at_least_one(radius: object) -> None:
    """v1-morphology acceptance 2-3: radius rejects non-int values and integers below one actionably."""
    source = _frame(np.zeros((2, 2, 1), dtype=np.float32), channels=["matte"])
    with pytest.raises(ValueError) as error:
        px.morphology.erosion(source, radius=radius)  # type: ignore[arg-type]
    _assert_actionable(error)


def test_morphology_shape_is_the_exact_disk_square_axis() -> None:
    """v1-morphology acceptance 2-3: shape accepts only case-sensitive disk and square tokens."""
    source = _frame(np.zeros((2, 2, 1), dtype=np.float32), channels=["matte"])
    for shape in SHAPES:
        assert px.morphology.dilation(source, radius=1, shape=shape).shape == source.shape
    with pytest.raises(ValueError) as error:
        px.morphology.dilation(source, radius=1, shape="circle")
    _assert_actionable(error)
    for shape in SHAPES:
        assert shape in str(error.value)


def test_hand_computed_radius_one_support_and_corner_borders() -> None:
    """v1-morphology acceptance 2, 4, and 7: hand-counted disk/square support fixes all border corners."""
    values = np.asarray([[5.0, 1.0, 9.0], [7.0, 3.0, 4.0], [8.0, 2.0, 6.0]], dtype=np.float32)[..., None]
    source = _frame(values, channels=["matte"])
    expected_disk_corner = {
        "mirror": (1.0, 7.0),
        "replicate": (1.0, 7.0),
        "wrap": (1.0, 9.0),
        "constant": (-2.0, 7.0),
    }
    for border, (expected_erode, expected_dilate) in expected_disk_corner.items():
        border_kwargs = {"border_value": -2.0} if border == "constant" else {}
        eroded = px.io.to_array(
            px.morphology.erosion(source, radius=1, border=border, **border_kwargs),
        ).get()
        dilated = px.io.to_array(
            px.morphology.dilation(source, radius=1, border=border, **border_kwargs),
        ).get()
        assert float(eroded[0, 0, 0]) == expected_erode
        assert float(dilated[0, 0, 0]) == expected_dilate

    disk = px.io.to_array(
        px.morphology.dilation(source, radius=1, shape="disk"),
    ).get()
    square = px.io.to_array(
        px.morphology.dilation(source, radius=1, shape="square"),
    ).get()
    assert float(disk[1, 1, 0]) == 7.0
    assert float(square[1, 1, 0]) == 9.0


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("border", BORDERS)
@pytest.mark.parametrize(("name", "dilate"), (("erosion", False), ("dilation", True)))
def test_morphology_primitives_match_independent_numpy_reference(
    name: str, dilate: bool, border: str, shape: str
) -> None:
    """v1-morphology acceptance 4 and 7: min/max match an independent NumPy scalar oracle for every token."""
    rng = np.random.default_rng(20260730)
    values = rng.uniform(-0.7, 1.8, size=(3, 4, 3)).astype(np.float32)
    source = _frame(values, colorspace="ACEScg", channels=["A", "custom", "Z"])
    border_value = -0.35
    expected = _primitive_reference(
        values,
        radius=2,
        shape=shape,
        border=border,
        border_value=border_value,
        dilate=dilate,
    )
    border_kwargs = {"border_value": border_value} if border == "constant" else {}
    result = getattr(px.morphology, name)(source, radius=2, shape=shape, border=border, **border_kwargs)
    np.testing.assert_array_equal(
        px.io.to_array(
            result,
        ).get(),
        expected,
    )


@pytest.mark.parametrize("name", MORPHOLOGY_NAMES[2:])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("border", BORDERS)
def test_morphology_compounds_match_the_defined_primitive_compositions(name: str, shape: str, border: str) -> None:
    """v1-morphology acceptance 8: open/close/differences match independently composed min/max references."""
    values = np.asarray(
        [[[0.2], [1.5], [-0.4]], [[0.8], [0.3], [1.2]], [[-0.2], [0.9], [0.1]]],
        dtype=np.float32,
    )
    source = _frame(values, channels=["matte"])
    border_value = 0.65
    expected = _compound_reference(
        values,
        operation=name,
        radius=1,
        shape=shape,
        border=border,
        border_value=border_value,
    )
    border_kwargs = {"border_value": border_value} if border == "constant" else {}
    result = getattr(px.morphology, name)(source, radius=1, shape=shape, border=border, **border_kwargs)
    np.testing.assert_array_equal(
        px.io.to_array(
            result,
        ).get(),
        expected,
    )


@pytest.mark.parametrize("name", MORPHOLOGY_NAMES)
@pytest.mark.parametrize("channel_count", (2, 5))
def test_morphology_all_operations_are_bit_exact_across_radius_shape_and_border(name: str, channel_count: int) -> None:
    """v1-morphology acceptance 2, 4, 7, and 8: every optimized path stays bit exact to the scalar oracle."""
    rng = np.random.default_rng(20260817)
    values = rng.uniform(-1.7, 2.3, size=(4, 5, channel_count)).astype(np.float32)
    source = _frame(values, colorspace="ACEScg", channels=[f"channel-{index}" for index in range(channel_count)])
    border_value = 0.375

    for radius in (1, 3, 6):
        for shape in SHAPES:
            for border in BORDERS:
                arguments = {
                    "radius": radius,
                    "shape": shape,
                    "border": border,
                    "border_value": border_value,
                }
                if name == "erosion":
                    expected = _primitive_reference(values, dilate=False, **arguments)
                elif name == "dilation":
                    expected = _primitive_reference(values, dilate=True, **arguments)
                else:
                    expected = _compound_reference(values, operation=name, **arguments)
                border_kwargs = {"border_value": border_value} if border == "constant" else {}
                result = getattr(px.morphology, name)(
                    source,
                    radius=radius,
                    shape=shape,
                    border=border,
                    **border_kwargs,
                )
                actual = px.io.to_array(result).get()
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("name", MORPHOLOGY_NAMES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("border", BORDERS)
def test_morphology_large_radius_fallback_is_bit_exact(name: str, shape: str, border: str) -> None:
    """v1-morphology acceptance 2, 4, 7, and 8: radii beyond the tiled budget retain exact behavior."""
    values = np.asarray([[[0.625]]], dtype=np.float32)
    source = _frame(values, colorspace="ACEScg", channels=["matte"])
    border_value = -0.375
    arguments = {"radius": 48, "shape": shape, "border": border, "border_value": border_value}
    if name == "erosion":
        expected = _primitive_reference(values, dilate=False, **arguments)
    elif name == "dilation":
        expected = _primitive_reference(values, dilate=True, **arguments)
    else:
        expected = _compound_reference(values, operation=name, **arguments)
    border_kwargs = {"border_value": border_value} if border == "constant" else {}
    result = getattr(px.morphology, name)(source, radius=48, shape=shape, border=border, **border_kwargs)
    actual = px.io.to_array(result).get()
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_replicate_default_is_neutral_for_uniform_frames() -> None:
    """v1-morphology acceptance 4 and 9: default replicate preserves uniform values at every edge."""
    values = np.full((2, 3, 2), (-0.5, 1.75), dtype=np.float32)
    source = _frame(values, channels=["negative", "highlight"])
    for name in MORPHOLOGY_NAMES[:2]:
        default = px.io.to_array(
            getattr(px.morphology, name)(source, radius=5),
        ).get()
        explicit = px.io.to_array(
            getattr(px.morphology, name)(source, radius=5, border="replicate"),
        ).get()
        np.testing.assert_array_equal(default, values)
        np.testing.assert_array_equal(default, explicit)


def test_morphology_preserves_metadata_channels_scene_values_and_input_privately() -> None:
    """v1-morphology acceptance 5-6: per-channel fp32 math preserves metadata, scene range, and input storage."""
    values = np.asarray(
        [[[-1.0, 2.0], [-0.5, 4.0], [-0.2, 3.0]], [[-0.8, 5.0], [-0.4, 6.0], [-0.1, 7.0]]],
        dtype=np.float32,
    )
    source = _frame(values, colorspace="ACEScg", gamma="logc4", channels=["A", "application-mask"])
    original = px.io.to_array(source, copy=True).get()
    result = px.morphology.dilation(source, radius=1, border="wrap")

    assert result.data.data.ptr != source.data.data.ptr
    assert (result.colorspace, result.gamma, result.channels) == ("ACEScg", "logc4", ("A", "application-mask"))
    np.testing.assert_array_equal(
        px.io.to_array(
            source,
        ).get(),
        original,
    )
    assert float(result.data.min()) < 0.0
    assert float(result.data.max()) > 1.0


@pytest.mark.parametrize("name", MORPHOLOGY_NAMES)
@pytest.mark.parametrize("dtype", ("float16", "uint8", "uint16"))
def test_morphology_rejects_non_fp32_frames_with_conversion_guidance(name: str, dtype: str) -> None:
    """v1-morphology acceptance 5: every public operation requires fp32 with an actionable cast path."""
    source = _frame(np.ones((2, 2, 1)), channels=["matte"], dtype=dtype)
    with pytest.raises(ValueError) as error:
        getattr(px.morphology, name)(source, radius=1)
    _assert_actionable(error)
    assert any(token in str(error.value) for token in ("cast_dtype", "recode_dtype", "dequantize"))


def test_morphology_border_and_border_value_follow_the_shared_contract() -> None:
    """v1-morphology acceptance 3-4: four borders and constant-only finite border_value fail fast symmetrically."""
    source = _frame(np.arange(6, dtype=np.float32).reshape(2, 3, 1), channels=["matte"])
    for border in BORDERS:
        kwargs = {"border_value": -1.25} if border == "constant" else {}
        assert px.morphology.erosion(source, radius=1, border=border, **kwargs).shape == source.shape

    with pytest.raises(ValueError) as error:
        px.morphology.erosion(source, radius=1, border="reflect")
    _assert_actionable(error)
    for token in BORDERS:
        assert token in str(error.value)

    for border_value in (None, True, "0", float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError) as error:
            px.morphology.erosion(source, radius=1, border="constant", border_value=border_value)  # type: ignore[arg-type]
        _assert_actionable(error)
    for border in BORDERS[:3]:
        with pytest.raises(ValueError) as error:
            px.morphology.erosion(source, radius=1, border=border, border_value=0.0)
        _assert_actionable(error)


def test_morphology_vocabulary_defines_shape_composites_and_border_default(vocabulary_markdown: str) -> None:
    """v1-morphology acceptance 10: vocabulary fixes shape support, composite meanings, and replicate border default."""
    shape_section = vocabulary_markdown.split("## morphology shape\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    border_section = vocabulary_markdown.split("## border\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    for required in (
        "disk",
        "square",
        "dx² + dy² <= radius²",
        "Chebyshev",
        "px.morphology.white_tophat",
        "px.morphology.black_tophat",
        "small bright details",
        "small dark details",
    ):
        assert required in shape_section
    for name in MORPHOLOGY_NAMES:
        assert name in border_section
    for required in ("replicate", "default", "min", "max", "neutral"):
        assert required in border_section
